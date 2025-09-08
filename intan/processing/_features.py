import numpy as np
from scipy.signal import hilbert
from typing import Callable, Dict, List, Union


# Deflationary orthogonality
def orthogonalize(W, wp, i):
    """
    Orthogonalizes the weight vector wp with respect to the first i columns of W.

    Parameters:
        W: Weight matrix of shape (n_features, n_features).
        wp: Weight vector to be orthogonalized of shape (n_features,).
        i: Index of the column in W to orthogonalize against.

    Returns:
        wp: Orthogonalized weight vector of shape (n_features,).
    """
    return wp - ((wp @ W[:i, :].T) @ W[:i, :])


# wp normalization
def normalize(wp):
    """
    Normalizes the weight vector wp.

    Parameters:
        wp: Weight vector to be normalized of shape (n_features,).

    Returns:
        wp: Normalized weight vector of shape (n_features,).
    """
    return wp / np.linalg.norm(wp)


def rectify(emg_data):
    """
    Rectifies EMG data by converting all values to their absolute values.

    Parameters:
        emg_data (numpy array): List of numpy arrays or pandas DataFrame items with filtered EMG data.

    Returns:
        rectified_data: List of rectified numpy arrays (same shape as input data).
    """
    return np.abs(emg_data)


def window_rms(emg_data, window_size=400, verbose=False):
    """
    Apply windowed RMS to each channel in the multichannel EMG data.

    Parameters:
        emg_data: Numpy array of shape (num_samples, num_channels).
        window_size: Size of the window for RMS calculation.
        verbose: Whether to print progress.

    Returns:
        Smoothed EMG data with windowed RMS applied to each channel (same shape as input).
    """
    if verbose: print(f"| Applying windowed RMS with window size {window_size}")
    num_channels, num_samples = emg_data.shape
    rms_data = np.zeros((num_channels, num_samples))

    for i in range(num_channels):
        rms_data[i, :] = window_rms_1D(emg_data[i, :], window_size)

    return rms_data


def window_rms_1D(signal, window_size):
    """
    Compute windowed RMS of the signal.

    Parameters:
        signal: Input EMG signal.
        window_size: Size of the window for RMS calculation.

    Returns:
        Windowed RMS signal.
    """
    return np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode='same'))


def calculate_rms(data, window_size, verbose=False):
    """
    Calculates RMS features for each channel using non-overlapping windows.

    Parameters:
        data: 2D numpy array of EMG data (channels, samples).
        window_size: Size of the window for RMS calculation.
        verbose: Whether to print progress.
    Returns:
        rms_features: 2D numpy array of RMS features (channels, windows).
    """
    if verbose:
        print("| Calculating RMS features...")
    n_channels, n_samples = data.shape
    n_windows = n_samples // window_size
    rms_features = np.zeros((n_channels, n_windows))

    for ch in range(n_channels):
        for i in range(n_windows):
            window = data[ch, i * window_size:(i + 1) * window_size]
            rms_features[ch, i] = np.sqrt(np.mean(window ** 2))

    return rms_features  # Shape (n_channels, n_windows)


def compute_rolling_rms(signal, window_size=50):
    """Compute rolling RMS on a 1D signal."""
    squared = np.square(signal)
    window = np.ones(window_size) / window_size
    return np.sqrt(np.convolve(squared, window, mode='same'))


def downsample(emg_data, sampling_rate, target_fs=1000):
    """
    Downsamples the EMG data to the target sampling rate.

    Parameters:
        emg_data: 2D numpy array of shape (num_channels, num_samples).
        sampling_rate: Sampling rate of the original EMG data.
        target_fs: Target sampling rate for downsampling.

    Returns:
        downsampled_data: 2D numpy array of shape (num_channels, downsampled_samples).
    """
    # Compute the downsampling factor
    downsample_factor = int(sampling_rate / target_fs)

    # Downsample the data by taking every nth sample
    downsampled_data = emg_data[:, ::downsample_factor]

    return downsampled_data


def common_average_reference(emg_data, ignore_channels=None):
    """
    Applies Common Average Referencing (CAR) to the multi-channel EMG data.

    Parameters:
        emg_data: 2D numpy array of shape (num_channels, num_samples).
        ignore_channels: List of channels to ignore in CAR calculation (optional).

    Returns:
        car_data: 2D numpy array after applying CAR (same shape as input).

    """
    bad_channels = []
    if ignore_channels is not None:
        bad_channels = ignore_channels

    good_channels = [i for i in range(emg_data.shape[0]) if i not in bad_channels]
    if not good_channels:
        raise ValueError("No good channels available for Common Average Referencing.")

    # Compute the common average (mean across all channels at each time point)
    mean_signal = np.mean(emg_data[good_channels, :], axis=0)  # Shape: (good_channels, n_samples)

    # Subtract the common average from each channel
    return emg_data - mean_signal  # Broadcast subtraction across channels


def envelope_extraction(data, method='hilbert'):
    """
    Extracts the envelope of the EMG signal using the Hilbert transform.

    Parameters:
        data: 2D numpy array of EMG data (channels, samples).
        method: Method for envelope extraction ('hilbert' or other).

    Returns:
        envelope: 2D numpy array of the envelope (channels, samples).
    """
    if method == 'hilbert':
        analytic_signal = hilbert(data, axis=1)
        envelope = np.abs(analytic_signal)
    else:
        raise ValueError("Unsupported method for envelope extraction.")
    return envelope


def z_score_norm(data):
    """
    Apply z-score normalization to the input data.

    Parameters:
        data: 2D numpy array of shape (channels, samples).

    Returns:
        normalized_data: 2D numpy array of shape (channels, samples) after z-score normalization.
    """
    mean = np.mean(data, axis=1)[:, np.newaxis]
    std = np.std(data, axis=1)[:, np.newaxis]
    normalized_data = (data - mean) / std
    return normalized_data


# RMS (Root Mean Square)
def compute_rms(emg_window, axis=-1):
    """
    Compute RMS of EMG data along a given axis.

    Parameters:
        emg_window (np.ndarray): EMG data. Can be 1D or 2D.
            - 1D shape: (n_samples,)
            - 2D shape: (n_channels, n_samples)
        axis (int): Axis to compute RMS over. Default is -1 (last axis).

    Returns:
        np.ndarray or float: RMS value(s) along the given axis.
            - If input is 1D: returns float
            - If input is 2D: returns 1D array (n_channels,)
    """
    emg_window = np.asarray(emg_window)
    return np.sqrt(np.mean(emg_window ** 2, axis=axis))


def compute_grid_average(emg_data, grid_spacing=8, axis=0):
    """
    Computes the average of the EMG grids according to the grid spacing. For example, a spacing of 8 means that
    channels 1, 9, 17, etc. will be averaged together to form the first grid, and so on.

    Parameters:
        emg_data (np.ndarray): 2D numpy array of shape (num_channels, num_samples).
        grid_spacing (int): Number of channels to average together.
        axis (int): Axis along which to compute the grid averages.

    Returns:
        grid_averages (np.ndarray): 2D numpy array of shape (num_grids, num_samples).
    """
    num_channels, num_samples = emg_data.shape
    num_grids = num_channels // grid_spacing
    grid_averages = np.zeros((num_grids, num_samples))

    for i in range(num_grids):
        start_idx = i * grid_spacing
        end_idx = (i + 1) * grid_spacing
        grid_averages[i, :] = np.mean(emg_data[start_idx:end_idx, :], axis=axis)

    return grid_averages


def variance(data):
    """
    Computes the variance of the input data.

    Args:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the variance.

    Returns:
        np.ndarray: Variance value for each channel.

    """
    return np.var(data)


def mean_absolute_value(data):
    """
    Computes the Mean Absolute Value (MAV) of the input data.

    Args:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the MAV.

    Returns:
        np.ndarray: MAV value for each channel.

    """
    return np.mean(np.abs(data))


def zero_crossings(ch, threshold=0.01):
    """
    Computes the number of zero crossings in the input data.

    Args:
        ch (np.ndarray): Input data for which to compute the zero crossings.
        threshold (float): Threshold for detecting significant changes.

    Returns:
        int: Number of zero crossings.

    """
    return np.sum((np.diff(np.sign(ch)) != 0) & (np.abs(np.diff(ch)) > threshold))


def integrated_emg(data):
    """
    Computes the Integrated EMG (IEMG) of the input data.

    Args:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the IEMG.

    Returns:
        np.ndarray: Integrated EMG value for each channel.

    """
    return np.sum(np.abs(data))


def slope_sign_changes(ch, threshold=0.01):
    """
    Computes the number of slope sign changes in the input data.

    Args:
        ch (np.ndarray): Input data for which to compute the slope sign changes.
        threshold (float): Threshold for detecting significant changes.

    Returns:
        int: Number of slope sign changes.

    """
    return np.sum((np.diff(np.sign(np.diff(ch))) != 0) & (np.abs(np.diff(np.diff(ch))) > threshold))


def waveform_length(data):
    """
    Computes the waveform length of the input data.

    Args:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the waveform length.

    Returns:
        np.ndarray: Waveform length for each channel.

    """
    return np.sum(np.abs(np.diff(data)))


def root_mean_square(data):
    """
    Computes the Root Mean Square (RMS) of the input data.

    Args:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the RMS.

    Returns:
        np.ndarray: RMS value for each channel.

    """
    return np.sqrt(np.mean(data ** 2))


FEATURE_REGISTRY = {
    'mean_absolute_value': mean_absolute_value,
    'zero_crossings': zero_crossings,
    'slope_sign_changes': slope_sign_changes,
    'waveform_length': waveform_length,
    'root_mean_square': root_mean_square,
    'variance': variance,
    'integrated_emg': integrated_emg,
}


def extract_features(segment, feature_fns=None):
    """
    Extracts features from a multichannel segment.

    ARgs:
        segment: np.ndarray (n_channels, n_samples)
        feature_fns: list of strings or callables

    Returns:
        1D np.ndarray: flattened feature vector

    """
    if feature_fns is None:
        feature_fns = list(FEATURE_REGISTRY.values())

    # Resolve string names to callables
    resolved_fns = []
    for fn in feature_fns:
        if isinstance(fn, str):
            if fn not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature name: {fn}")
            resolved_fns.append(FEATURE_REGISTRY[fn])
        else:
            resolved_fns.append(fn)

    feats = []
    for ch in segment:
        for fn in resolved_fns:
            feats.append(fn(ch))
    return np.array(feats)


def extract_features_sliding_window(data: np.ndarray, fs: float, window_ms: float, step_ms: float, feature_fns=None) -> np.ndarray:
    """
    Args:
        data: (n_channels, n_samples) array of EMG data
        fs:  (int)  sampling rate in Hz
        window_ms: window length in ms
        step_ms:   step between windows in ms

    Returns:
        (n_windows, n_features)
    """
    # compute sizes in samples
    w = int(window_ms/1000 * fs)
    s = int(step_ms/1000 * fs)
    n_samples = data.shape[1]

    # figure out how many windows (floor)
    n_windows = 1 + (n_samples - w)//s

    if feature_fns is None:
        feature_fns = list(FEATURE_REGISTRY.values())

    # Resolve string names to callables
    resolved_fns = []
    for fn in feature_fns:
        if isinstance(fn, str):
            if fn not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature name: {fn}")
            resolved_fns.append(FEATURE_REGISTRY[fn])
        else:
            resolved_fns.append(fn)

    feats = []
    for i in range(n_windows):
        start = i * s
        seg = data[:, start:start + w]
        # returns vector of length n_features
        fv = extract_features(seg, resolved_fns)
        feats.append(fv)

    return np.vstack(feats)  # shape (n_windows, n_features)


def feature_spec_from_registry(
    feature_registry: Dict[str, Callable],
    feature_fns: List[Union[str, Callable]] | None = None,
    *,
    per_channel: bool = True,
    layout: str = "channel_major",   # [ch0 feats..., ch1 feats..., ...]
    channels: str = "training_order" # clarifies feature vector is built in training channel order
) -> Dict:
    """
    Build a self-describing spec for your feature vector.

    If feature_fns is None, we assume 'use all' features in the registry, in the
    registry's key order.
    """
    if feature_fns is None:
        names = list(feature_registry.keys())
    else:
        names = []
        for fn in feature_fns:
            if isinstance(fn, str):
                names.append(fn)
            else:
                names.append(getattr(fn, "__name__", str(fn)))

    # In your current registry, each function returns a scalar per-channel.
    dims_per_feature = {name: 1 for name in names}

    return {
        "per_channel": bool(per_channel),
        "order": names,                 # feature names in the order they were concatenated
        "dims_per_feature": dims_per_feature,
        "layout": layout,
        "channels": channels            # how channels were ordered when flattening
    }
