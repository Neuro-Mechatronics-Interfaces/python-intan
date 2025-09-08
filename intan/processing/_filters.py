"""
nml_hand_exo.processing._filters

Comprehensive EMG signal preprocessing module.

Includes:
- Bandpass, lowpass, and notch filters
- Hilbert envelope extraction
- RMS and windowed RMS computation
- Common average referencing (CAR)
- Sliding windows and PCA-based dimensionality reduction
- CNN-ECA compatible preprocessing pipeline

This module supports feature extraction pipelines for real-time classification
and pre-training EMG datasets with overlapping or fixed windows.
"""

import time
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, iirnotch, lfilter, lfilter_zi
from ._features import compute_rms, window_rms, common_average_reference, envelope_extraction

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

from scipy.signal import butter, filtfilt, iirnotch



def preprocess_emg(emg_data, sample_rate):
    """
    Applies filtering and extracts RMS features.

    Parameters:
        emg_data: 2D numpy array of EMG data (channels, samples).
        sample_rate: Sampling rate of the EMG data.

    Returns:
        rms_features: 2D numpy array of RMS features (channels, windows).
    """
    filtered_data = notch_filter(emg_data, fs=sample_rate, f0=60)
    filtered_data = bandpass_filter(filtered_data, lowcut=20, highcut=400, fs=sample_rate, order=2, axis=1)
    rms_features = compute_rms(filtered_data, int(0.1 * sample_rate))
    return rms_features


def parse_channel_ranges(channel_arg):
    """
    Parses a channel range string (e.g., [1:8, 64:72]) and returns a flat list of integers.

    Parameters:
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
            # channel_list.extend(range(start - 1, end))  # Convert to 0-based indexing
            channel_list.extend(range(start, end))
        else:
            # channel_list.append(int(r) - 1)  # Convert single channel to 0-based indexing
            channel_list.append(int(r))
    return channel_list


def notch_filter(data, fs=4000, f0=60.0, Q=10, axis=1):
    """
    Applies a notch filter to the data to remove 60 Hz interference. Assumes data shape (n_channels, n_samples).
    A bandwidth of 10 Hz is recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Parameters:
        data (ndarray): Input data to be filtered.
        fs (float): Sampling frequency of the data.
        f0 (float): Frequency to be removed from the data (60 Hz).
        Q (float): Quality factor of the notch filter.

    Returns:
        nn.array:

    Example:
        out = notch_filter(signal_in, 30000, 60, 10);
    """
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=axis)


def lowpass_filter(data, cutoff, fs, order=4, axis=1):
    """
    Applies a lowpass filter to the data using a Butterworth filter.

    Parameters:
        data (ndarray): Input data to be filtered.
        cutoff (float): Cutoff frequency.
        fs (float): Sampling frequency of the data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.

    Returns:
        ndarray: Filtered data.
    """
    b, a = butter(order, cutoff, btype="low", fs=fs)
    y = filtfilt(b, a, data, axis=axis)
    return y


def bandpass_filter(data, lowcut=10, highcut=500, fs=4000, order=4, axis=1, verbose=False):
    """
    Applies a bandpass filter to the data using a Butterworth filter.

    Parameters:
        data (ndarray): Input data to be filtered.
        lowcut (float): Low cutoff frequency.
        highcut (float): High cutoff frequency.
        fs (float): Sampling frequency of the data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.
        verbose (bool): Whether to print filter parameters.

    Returns:
        ndarray: Filtered data.
    """
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    y = filtfilt(b, a, data, axis=axis)
    return y


def filter_emg(emg_data, filter_type='bandpass', lowcut=30, highcut=500, fs=1259, order=5, verbose=False):
    """
    Applies a bandpass or lowpass filter to EMG data using numpy arrays.

    Parameters:
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
        filtered_data = bandpass_filter(emg_data, lowcut, highcut, fs, order, axis=0)
    elif filter_type == 'lowpass':
        if verbose: print(f"| Applying butterworth lowpass filter: {lowcut} Hz {order} order")
        filtered_data = lowpass_filter(emg_data, lowcut, fs, order, axis=0)

    toc = time.process_time()
    if verbose:
        print(f"| | Filtering time = {1000 * (toc - tic):.2f} ms")

    # Convert list of arrays to a single 2D numpy array
    filtered_data = np.stack(filtered_data, axis=0)  # Stack along axis 0 (channels)

    return filtered_data


def process_emg_pipeline(data, lowcut=30, highcut=500, order=5, window_size=400, verbose=False):
    """
    Processing steps to match the CNN-ECA methodology
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10669079/
    Input data is assumed to have shape (N_channels, N_samples)

    Parameters:
        data: 2D numpy array of EMG data (channels, samples).
        lowcut: Low cutoff frequency for the bandpass filter.
        highcut: High cutoff frequency for the bandpass filter.
        order: Order of the Butterworth filter.
        window_size: Window size for RMS calculation.
        verbose: Whether to print progress.

    Returns:
        smoothed: 2D numpy array of processed EMG data (channels, samples).
    """
    emg_data = data['amplifier_data']  # Extract EMG data
    sample_rate = int(data['frequency_parameters']['board_dig_in_sample_rate'])  # Extract sampling rate

    # Overwrite the first and last second of the data with 0 to remove edge effects
    # emg_data[:, :sample_rate] = 0.0
    emg_data[:, -sample_rate:] = 0.0  # Just first second

    # Apply bandpass filter
    bandpass_filtered = filter_emg(emg_data, 'bandpass', lowcut, highcut, sample_rate, order)

    # Rectify
    # rectified = rectify_emg(bandpass_filtered)
    rectified = bandpass_filtered

    # Apply Smoothing
    # smoothed = window_rms(rectified, window_size=window_size)
    smoothed = envelope_extraction(rectified, method='hilbert')

    return smoothed


def sliding_window(data, window_size, step_size):
    """
    Splits the data into overlapping windows.

    Parameters:
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




# def apply_pca(data, num_components=8, verbose=False):
#     """
#     Applies PCA to reduce the number of EMG channels to the desired number of components.
#
#     Parameters:
#         data: 2D numpy array of EMG data (channels, samples) -> (128, 500,000).
#         num_components: Number of principal components to reduce to (e.g., 8).
#
#     Returns:
#         pca_data: 2D numpy array of reduced EMG data (num_components, samples).
#         explained_variance_ratio: Percentage of variance explained by each of the selected components.
#     """
#     # Step 1: Standardize the data across the channels
#     scaler = StandardScaler()
#     features_std = scaler.fit_transform(data)  # Standardizing along the channels
#
#     # Step 2: Apply PCA
#     pca = PCA(n_components=num_components)
#     pca_data = pca.fit_transform(features_std)  # Apply PCA on the transposed data
#
#     if verbose:
#         print("Original shape:", data.shape)
#         print("PCA-transformed data shape:", pca_data.shape)
#
#     # Step 3: Get the explained variance ratio (useful for understanding how much variance is retained)
#     explained_variance_ratio = pca.explained_variance_ratio_
#
#     return pca_data, explained_variance_ratio


