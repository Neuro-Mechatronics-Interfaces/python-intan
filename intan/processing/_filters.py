""" This module contains utility functions for processing EMG data.

"""
import os
import time
#import pywt
#import pathlib
import numpy as np
#import pandas as pd
from scipy.signal import find_peaks, peak_widths, butter, filtfilt, hilbert, iirnotch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define constants for windowing parameters and feature extraction
#WINDOW_SIZE = 400
#OVERLAP = 200

# Definition of G'
# def Gprime(x):
#     return np.tanh(x)
#
#
# # Definition of G''
# def Gsecond(x):
#     return np.ones(x.shape) - np.power(np.tanh(x), 2)
#
#
# # Center matrix X
# def centerMatrix(X, N):
#     mean = X.mean(axis=1)
#     M = X - (mean.reshape((N, 1)) @ np.ones((1, X.shape[1])))
#     return M
#
#
# # Whiten matrix X with eigenvalue decomposition
# def whitenMatrix(X):
#     D, E = np.linalg.eigh(X @ X.T)
#     DE = np.diag(1 / np.sqrt(D + 1e-5)) @ E.T
#
#     return DE @ X
#
#
# # One-unit algorithm step
# def oneUnit(X, wp):
#     term1 = np.mean((X @ Gprime(wp @ X).T), axis=1)
#     term2 = np.mean(Gsecond(wp @ X), axis=1) * wp
#     return term1 - term2


# Deflationary orthogonalization
def orthogonalize(W, wp, i):
    return wp - ((wp @ W[:i, :].T) @ W[:i, :])


# wp normalization
def normalize(wp):
    return wp / np.linalg.norm(wp)

#
# def fastICA(X, C, max_iter):
#     N = X.shape[0]
#     X = centerMatrix(X, N)
#     X = whitenMatrix(X)
#
#     W = np.zeros((C, N))
#     for i in range(C):
#         wp = np.random.rand(1, N)
#         for _ in range(max_iter):
#             wp = oneUnit(X, wp)
#             wp = orthogonalize(W, wp, i)
#             wp = normalize(wp)
#             W[i, :] = wp
#
#     return W @ X

#
#
# def process_emg_files(file_paths, metrics_data, gesture_map):
#     """Processes each EMG recording file and extracts RMS features."""
#     X_list, y_list = [], []
#
#     for file in file_paths:
#         # Load EMG Data
#         result, data_present = rhd_utils.load_file(file, verbose=False)
#         if not data_present:
#             continue
#
#         emg_data = result['amplifier_data']
#         sample_rate = int(result['frequency_parameters']['board_dig_in_sample_rate'])
#
#         # Preprocess the EMG signal
#         rms_features = preprocess_emg(emg_data, sample_rate)
#
#         # Retrieve gesture label
#         file_name = os.path.basename(file)
#         if file_name not in metrics_data['File Name'].values:
#             print(f"⚠️ Warning: No entry found for {file_name} in metrics data. Skipping.")
#             continue
#
#         gesture = metrics_data[metrics_data['File Name'] == file_name]['Gesture'].values[0]
#
#         # Append to lists
#         X_list.append(rms_features.T)  # Shape (N_samples, 128)
#         y_list.append(np.full(rms_features.shape[1], gesture_map[gesture]))
#
#     return X_list, y_list

def preprocess_emg(emg_data, sample_rate):
    """Applies filtering and extracts RMS features."""
    filtered_data = notch_filter(emg_data, fs=sample_rate, f0=60)
    filtered_data = butter_bandpass_filter(filtered_data, lowcut=20, highcut=400, fs=sample_rate, order=2, axis=1)
    rms_features = calculate_rms(filtered_data, int(0.1 * sample_rate))
    return rms_features

def parse_channel_ranges(channel_arg):
    """
    Parses a channel range string (e.g., [1:8, 64:72]) and returns a flat list of integers.

    Args:
        channel_arg (str): The string containing channel ranges (e.g., "[1:8, 64:72]").

    Returns:
        list: A flat list of integers.
    """
    # Remove square brackets and split by commas
    channel_arg = channel_arg.strip("[]")
    ranges = channel_arg.split(",")

    channel_list = []
    for r in ranges:
        if ":" in r:
            start, end = map(int, r.split(":"))
            #channel_list.extend(range(start - 1, end))  # Convert to 0-based indexing
            channel_list.extend(range(start, end))
        else:
            #channel_list.append(int(r) - 1)  # Convert single channel to 0-based indexing
            channel_list.append(int(r))
    return channel_list

#
# # Define feature extraction functions
# def extract_wavelet_features(emg_data, window_size=WINDOW_SIZE, overlap=OVERLAP):
#     features = []
#     num_samples, num_channels = emg_data.shape
#     print(f"Num samples: {num_samples}, Num channels: {num_channels}")
#     step = window_size - overlap
#
#     for start in range(0, num_samples - window_size + 1, step):
#         window = emg_data[start:start + window_size]
#         window_features = []
#
#         for channel_data in window.T:
#             # Apply 2-level wavelet decomposition using dbl mother wavelet
#             coeffs = pywt.wavedec(channel_data, 'db4', level=2)
#
#             # Extract 19 statistical features from the detail and approximation coefficients
#             # Or pick and choose the features you want to extract
#             for coeff in coeffs:
#                 window_features.extend([
#                     np.sum(np.abs(coeff)),  # IEMG
#                     np.mean(np.abs(coeff)),  # MAV
#                     np.sum(coeff ** 2),  # SSI
#                     np.sqrt(np.mean(coeff ** 2)),  # RMS
#                     np.var(coeff),  # VAR
#                     np.mean(coeff > 0),  # MYOP
#                     np.sum(np.abs(np.diff(coeff))),  # WL
#                     np.mean(np.abs(np.diff(coeff))),  # DAMV
#                     np.sum(coeff ** 2) / len(coeff),  # Second-order moment (M2)
#                     np.var(np.diff(coeff)),  # DVARV
#                     np.std(np.diff(coeff)),  # DASDV
#                     np.sum(np.abs(coeff) > 0.05),  # WAMP (threshold = 0.05)
#                     np.sum(np.abs(np.diff(coeff, 2))),  # IASD
#                     np.sum(np.abs(np.diff(coeff, 3))),  # IATD
#                     np.sum(np.exp(np.abs(coeff))),  # IEAV
#                     np.sum(np.log(np.abs(coeff) + 1e-6)),  # IALV
#                     np.sum(np.exp(coeff)),  # IE
#                     np.min(coeff),  # MIN
#                     np.max(coeff)  # MAX
#                 ])
#         features.append(window_features)
#
#     return np.array(features)

def notch_filter(data, fs=4000, f0=60.0, Q=30):
    """Applies a notch filter to the data to remove 60 Hz interference.
        Assumes data shape (n_channels, n_samples).
    """
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=1)

def butter_bandpass(lowcut, highcut, fs, order=5):
    # butterworth bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, axis=0):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data, axis=axis)  # Filter along axis 0 (time axis) for all channels simultaneously
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0, verbose=False):
    if verbose:
        print(f"| Applying butterworth bandpass filter: {lowcut}-{highcut} Hz {order} order")
    # function to implement filter on data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)  # Filter channels simultaneously
    return y


def filter_emg(emg_data, filter_type='bandpass', lowcut=30, highcut=500, fs=1259, order=5, verbose=False):
    """
    Applies a bandpass or lowpass filter to EMG data using numpy arrays.

    Args:
        emg_data: Numpy array of shape (num_samples, num_channels) with EMG data.
        filter_type: Type of filter to apply ('bandpass' or 'lowpass').
        lowcut: Low cutoff frequency for the bandpass filter.
        highcut: High cutoff frequency for the bandpass filter.
        fs: Sampling rate of the EMG data.
        order: Filter order.
        verbose: Whether to print progress.

    Returns:
        Filtered data as a numpy array (same shape as input data).
    """
    tic = time.process_time()

    if filter_type == 'bandpass':
        if verbose: print(f"| Applying butterworth bandpass filter: {lowcut}-{highcut} Hz {order} order")
        filtered_data = butter_bandpass_filter(emg_data, lowcut, highcut, fs, order)
    elif filter_type == 'lowpass':
        if verbose: print(f"| Applying butterworth lowpass filter: {lowcut} Hz {order} order")
        filtered_data = butter_lowpass_filter(emg_data, lowcut, fs, order)

    toc = time.process_time()
    if verbose:
        print(f"| | Filtering time = {1000 * (toc - tic):.2f} ms")

    # Convert list of arrays to a single 2D numpy array
    filtered_data = np.stack(filtered_data, axis=0)  # Stack along axis 0 (channels)

    return filtered_data


def rectify(emg_data):
    """
    Rectifies EMG data by converting all values to their absolute values.

    Args:
        EMGDataDF: List of numpy arrays or pandas DataFrame items with filtered EMG data.

    Returns:
        rectified_data: List of rectified numpy arrays (same shape as input data).
    """
    rectified_data = np.abs(emg_data)

    return rectified_data


def window_rms(emg_data, window_size=400, verbose=False):
    """
    Apply windowed RMS to each channel in the multi-channel EMG data.

    Args:
        emg_data: Numpy array of shape (num_samples, num_channels).
        window_size: Size of the window for RMS calculation.

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

    Args:
        signal: Input EMG signal.
        window_size: Size of the window for RMS calculation.

    Returns:
        Windowed RMS signal.
    """
    return np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode='same'))



def calculate_rms(data, window_size, verbose=False):
    """Calculates RMS features for each channel using non-overlapping windows."""
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

def downsample(emg_data, sampling_rate, target_fs=1000):
    """
    Downsamples the EMG data to the target sampling rate.

    Args:
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

def common_average_reference(emg_data, verbose=False):
    """
    Applies Common Average Referencing (CAR) to the multi-channel EMG data.

    Args:
        emg_data: 2D numpy array of shape (num_channels, num_samples).

    Returns:
        car_data: 2D numpy array after applying CAR (same shape as input).
    """
    if verbose:
        print("| Subtracting common average reference")
    # Compute the common average (mean across all channels at each time point)
    common_avg = np.mean(emg_data, axis=0)  # Shape: (num_samples,)

    # Subtract the common average from each channel
    car_data = emg_data - common_avg  # Broadcast subtraction across channels

    return car_data


def envelope_extraction(data, method='hilbert'):
    if method == 'hilbert':
        analytic_signal = hilbert(data, axis=1)
        envelope = np.abs(analytic_signal)
    else:
        raise ValueError("Unsupported method for envelope extraction.")
    return envelope


def process_emg_pipeline(data, lowcut=30, highcut=500, order=5, window_size=400, verbose=False):
    # Processing steps to match the CNN-ECA methodology
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC10669079/
    # Input data is assumed to have shape (N_channels, N_samples)

    emg_data = data['amplifier_data']  # Extract EMG data
    sample_rate = int(data['frequency_parameters']['board_dig_in_sample_rate'])  # Extract sampling rate

    # Overwrite the first and last second of the data with 0 to remove edge effects
    #emg_data[:, :sample_rate] = 0.0
    emg_data[:, -sample_rate:] = 0.0  # Just first second

    # Apply bandpass filter
    bandpass_filtered = filter_emg(emg_data, 'bandpass', lowcut, highcut, sample_rate, order)

    # Rectify
    #rectified = rectify_emg(bandpass_filtered)
    rectified = bandpass_filtered

    # Apply Smoothing
    #smoothed = window_rms(rectified, window_size=window_size)
    smoothed = envelope_extraction(rectified, method='hilbert')

    return smoothed


def sliding_window(data, window_size, step_size):
    """
    Splits the data into overlapping windows.

    Args:
        data: 2D numpy array of shape (channels, samples).
        window_size: Window size in number of samples.
        step_size: Step size in number of samples.

    Returns:
        windows: List of numpy arrays, each representing a window of data.
    """
    num_channels, num_samples = data.shape
    windows = []

    for start in range(0, num_samples - window_size + 1, step_size):
        window = data[:, start:start + window_size]
        windows.append(window)

    return windows


def apply_pca(data, num_components=8, verbose=False):
    """
    Applies PCA to reduce the number of EMG channels to the desired number of components.

    Args:
        data: 2D numpy array of EMG data (channels, samples) -> (128, 500,000).
        num_components: Number of principal components to reduce to (e.g., 8).

    Returns:
        pca_data: 2D numpy array of reduced EMG data (num_components, samples).
        explained_variance_ratio: Percentage of variance explained by each of the selected components.
    """
    # Step 1: Standardize the data across the channels
    scaler = StandardScaler()
    features_std = scaler.fit_transform(data)  # Standardizing along the channels

    # Step 2: Apply PCA
    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(features_std) # Apply PCA on the transposed data

    if verbose:
        print("Original shape:", data.shape)
        print("PCA-transformed data shape:", pca_data.shape)

    # Step 3: Get the explained variance ratio (useful for understanding how much variance is retained)
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_data, explained_variance_ratio


# def apply_gesture_label(df, sampling_rate, data_metrics, start_index_name='Start Index', n_trials_name='N_trials', trial_interval_name='Trial Interval (s)', gesture_name='Gesture'):
#     """ Applies the Gesture label to the dataframe and fills in the corresponding gesture labels for samples in the
#     dataframe. The gesture labels are extracted from the data_metrics dataframe.
#     """
#
#     # Initialize a label column in the dataframe
#     #df['Gesture'] = 'Rest'  # Default is 'Rest'
#
#     # Collect the data metrics for the current file
#     start_idx = data_metrics[start_index_name]
#     print(f"Start index: {start_idx}")
#     n_trials = data_metrics[n_trials_name]
#     print(f"Number of trials: {n_trials}")
#     trial_interval = data_metrics[trial_interval_name]
#     print(f"Trial interval: {trial_interval}")
#     gesture = data_metrics[gesture_name]
#     print(f"Gesture: {gesture}")
#
#     # Iterate over each trial and assign the gesture label to the corresponding samples
#     for i in range(n_trials):
#         # Get start and end indices for the flex (gesture) and relax
#         start_flex = start_idx + i * sampling_rate * trial_interval
#         end_flex = start_flex + sampling_rate * trial_interval / 2  # Flex is half of interval
#
#         # Label the flex periods as the gesture
#         df.loc[start_flex:end_flex, 'Gesture'] = gesture
#
#     return df

def z_score_norm(data):
    """
    Apply z-score normalization to the input data.

    Args:
        data: 2D numpy array of shape (channels, samples).

    Returns:
        normalized_data: 2D numpy array of shape (channels, samples) after z-score normalization.
    """
    mean = np.mean(data, axis=1)[:, np.newaxis]
    std = np.std(data, axis=1)[:, np.newaxis]
    normalized_data = (data - mean) / std
    return normalized_data


# RMS (Root Mean Square)
def compute_rms(emg_window):
    return np.sqrt(np.mean(emg_window**2))

# WL (Waveform Length)
# def compute_wl(emg_window):
#     return np.sum(np.abs(np.diff(emg_window)))
#
# # MAS (Median Amplitude Spectrum)
# def compute_mas(emg_window):
#     fft_values = np.fft.fft(emg_window)
#     magnitude_spectrum = np.abs(fft_values)
#     return np.median(magnitude_spectrum)

# SampEn (Sample Entropy)
#import antropy as ant

# def compute_sampen(emg_window, m=2, r=0.2):
#     return ant.sample_entropy(emg_window, order=m)

# Extract features for a given EMG window
# def extract_features(emg_window):
#     """https://link.springer.com/article/10.1007/s00521-019-04142-8"""
#     features = [
#         compute_rms(emg_window),
#         compute_wl(emg_window),
#         compute_mas(emg_window),
#         compute_sampen(emg_window)
#     ]
#     return np.array(features)
#
# def create_lagged_features(features, n_lags=4, verbose=False):
#     """
#     Create lagged features by concatenating the current bin with the previous n_lags bins.
#     Args:
#         features: input array of shape (n_channels, n_bins).
#         n_lags:  Number of previous bins to concatenate with the current bin.
#
#     Returns:
#         lagged_features: array of shape (n_bins - n_lags, n_channels * (n_lags + 1)).
#     """
#     if verbose: print(f"| Creating lagged features with {n_lags} lags...")
#     num_bins = features.shape[1]  # Total number of bins
#     lagged_features = []
#     for i in range(4, num_bins):
#         # Concatenate the current bin with the previous 4 bins
#         current_features = features[:, (i - 4):i + 1].flatten()  # Flatten to create feature vector
#         lagged_features.append(current_features)
#
#     if verbose: print(f"| Lagged features shape: {np.array(lagged_features).shape}")
#     return np.array(lagged_features)

def compute_grid_average(emg_data, grid_spacing=8, axis=0):
    """Function that computes the average of the EMG grids according to the grid spacing. For example, a spacing of 8 means that
    channels 1, 9, 17, etc. will be averaged together to form the first grid, and so on.

    Args:
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