""" This module contains utility functions for processing EMG data.

"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, butter, filtfilt, hilbert
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_config_file(config_file):
    # Dictionary to store the key-value pairs
    config_data = {}

    # Open the CONFIG.txt file and read its contents
    with open(config_file, 'r') as file:
        for line in file:
            # Strip whitespace and ignore empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value at the first '='
            key, value = line.split('=', 1)
            config_data[key.strip()] = value.strip()

    return config_data

def get_metrics_file(metrics_filepath, verbose=False):
    if os.path.isfile(metrics_filepath):
        if verbose:
            print("Metrics file found.")
        return pd.read_csv(metrics_filepath)
    else:
        print(f"Metrics file not found: {metrics_filepath}. Please correct file path or generate the metrics file.")
        return None


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


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data, axis=0)  # Filter along axis 0 (time axis) for all channels simultaneously
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # function to implement filter on data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)  # Filter along axis 0 (time axis) for all channels simultaneously
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
        filtered_data = butter_bandpass_filter(emg_data, lowcut, highcut, fs, order)
    elif filter_type == 'lowpass':
        filtered_data = butter_lowpass_filter(emg_data, lowcut, fs, order)

    toc = time.process_time()
    if verbose:
        print(f"Filtering time = {1000 * (toc - tic):.2f} ms")

    # Convert list of arrays to a single 2D numpy array
    filtered_data = np.stack(filtered_data, axis=0)  # Stack along axis 0 (channels)

    return filtered_data


def rectify_emg(emg_data):
    """
    Rectifies EMG data by converting all values to their absolute values.

    Args:
        EMGDataDF: List of numpy arrays or pandas DataFrame items with filtered EMG data.

    Returns:
        rectified_data: List of rectified numpy arrays (same shape as input data).
    """
    rectified_data = np.abs(emg_data)

    return rectified_data


def window_rms(emg_data, window_size=400):
    """
    Apply windowed RMS to each channel in the multi-channel EMG data.

    Args:
        emg_data: Numpy array of shape (num_samples, num_channels).
        window_size: Size of the window for RMS calculation.

    Returns:
        Smoothed EMG data with windowed RMS applied to each channel (same shape as input).
    """
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


def common_average_reference(emg_data):
    """
    Applies Common Average Referencing (CAR) to the multi-channel EMG data.

    Args:
        emg_data: 2D numpy array of shape (num_channels, num_samples).

    Returns:
        car_data: 2D numpy array after applying CAR (same shape as input).
    """
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


def apply_pca(data, num_components=8, verbose=True):
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
    data_standardized = scaler.fit_transform(data.T).T  # Standardizing along the channels

    # Step 2: Apply PCA
    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(data_standardized.T).T  # Apply PCA on the transposed data

    if verbose:
        print("Original shape:", data.shape)
        print("PCA-transformed data shape:", pca_data.shape)

    # Step 3: Get the explained variance ratio (useful for understanding how much variance is retained)
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_data, explained_variance_ratio


def apply_gesture_label(df, sampling_rate, data_metrics, start_index_name='Start Index', n_trials_name='N_trials', trial_interval_name='Trial Interval (s)', gesture_name='Gesture'):
    """ Applies the Gesture label to the dataframe and fills in the corresponding gesture labels for samples in the
    dataframe. The gesture labels are extracted from the data_metrics dataframe.
    """

    # Initialize a label column in the dataframe
    df['Gesture'] = 'Rest'  # Default is 'Rest'

    # Collect the data metrics for the current file
    start_idx = data_metrics[start_index_name]
    n_trials = data_metrics[n_trials_name]
    trial_interval = data_metrics[trial_interval_name]
    gesture = data_metrics[gesture_name]

    # Iterate over each trial and assign the gesture label to the corresponding samples
    for i in range(n_trials):
        # Get start and end indices for the flex (gesture) and relax
        start_flex = start_idx + i * sampling_rate * trial_interval
        end_flex = start_flex + sampling_rate * trial_interval / 2  # Flex is half of interval

        # Label the flex periods as the gesture
        df.loc[start_flex:end_flex, 'Gesture'] = gesture

    return df


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
