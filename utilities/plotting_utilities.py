""" Plotting functions for visualizing EMG data and features.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utilities.rhd_utilities as rhd_utils
import pywt

def waterfall_plot(result, channel_indices, time_vector, plot_title=""):
    """
    Creates a waterfall plot for the specified channels with a user-defined title,
    custom color styling, and scale bars for time and voltage.

    Args:
        result: The dictionary containing the loaded RHD data.
        channel_indices: The indices of the channels to be plotted.
        time_vector: The time vector for the x-axis.
        plot_title: (Optional) Title for the plot provided by the user.
    """
    fig, ax = plt.subplots(figsize=(10, 12))

    offset = 0  # Start with no offset
    offset_increment = 200  # Increment for each row (adjust this based on data scale)
    cmap = plt.get_cmap('rainbow')  # You can also experiment with other color maps like 'jet', 'viridis', etc.
    num_channels = len(channel_indices)

    for i, channel_idx in enumerate(channel_indices):
        channel_data = result['amplifier_data'][channel_idx, :]
        # Use colormap to assign a color based on channel index
        color = cmap(i / num_channels)
        ax.plot(time_vector, channel_data + offset, color=color, linewidth=0.2)
        offset += offset_increment

    # Labeling and visualization
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Amplitude + Offset')
    ax.set_title(plot_title, fontsize=14, fontweight='bold')

    # Custom scale bar
    add_scalebars(ax)

    # Turn off x and y axis numbers
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the black box around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Insert text labels for channel 0 and channel 128
    insert_channel_labels(ax, time_vector, num_channels)

    # Insert vertical labels for "Extensor" and "Flexor"
    insert_vertical_labels(ax)

    plt.show()

def add_scalebars(ax):
    """
    Adds time and voltage scale bars to the plot in the lower left corner.
    """
    # Define position and length for the scale bars
    time_scale_length = 5  # 5 seconds
    voltage_scale_length = 10  # 10 mV converted to 10000 µV (since the data is in µV)

    # Add horizontal time scale bar (5 sec)
    ax.plot([0, time_scale_length], [-1000, -1000], color='gray', lw=3)
    ax.text(time_scale_length / 2, -1500, '5 sec', va='center', ha='center', fontsize=12, color='gray')

    # Add vertical voltage scale bar (10 mV)
    ax.plot([0, 0], [-1000, -1000 + voltage_scale_length * 10], color='gray', lw=3)
    ax.text(-0.5, -500, '10 mV', va='center', ha='center', rotation='vertical', fontsize=12, color='gray')
    #ax.text(-0.5, voltage_scale_length / 2 - 1000, '10 mV', va='center', ha='center', rotation='vertical', fontsize=12, color='gray')


def insert_channel_labels(ax, time_vector, num_channels):
    """
    Inserts text labels to indicate specific channels on the plot.

    Args:
        ax: The plot axes to add the text to.
        time_vector: The time vector for placing the text appropriately.
        num_channels: Total number of channels being plotted.
    """
    # Position the text for Channel 0 (near the bottom)
    x_pos = time_vector[-1]  # Position at the end of the time range
    ax.text(x_pos + 1, 200, 'Channel 0', fontsize=8, va='center', ha='left', color='black', fontweight='bold')

    # Position the text for Channel 128 (near the top)
    ax.text(x_pos + 1, 25500, 'Channel 128', fontsize=8, va='center', ha='left', color='black', fontweight='bold')


def insert_vertical_labels(ax):
    """
    Inserts vertical labels "Extensor" and "Flexor" for the two groups of channels.

    Args:
        ax: The plot axes to add the text to.
    """
    # Insert "Extensor" label vertically for the first 64 channels (left side)
    ax.text(-1, 5000, 'Extensor', fontsize=12, va='center', ha='center', color='black', rotation='vertical',
            fontweight='bold')

    # Insert "Flexor" label vertically for channels 65-128 (left side)
    ax.text(-1, 22000, 'Flexor', fontsize=12, va='center', ha='center', color='black', rotation='vertical',
            fontweight='bold')


def plot_time_domain_features(emg_signal, sample_rate=4000, window_size=400, overlap=200):
    """
    Plots time-domain features of EMG signals over time.

    Args:
        emg_signal: Raw EMG signal (1D array).
        sample_rate: Sampling rate of the EMG signal (default: 4000 Hz).
        window_size: Number of samples per window (default: 400).
        overlap: Number of overlapping samples (default: 200).
    """
    # Preprocess EMG signal (windowing)
    windows = rhd_utils.window_data(emg_signal, window_size=window_size, overlap=overlap)

    # Extract time-domain features for each window
    time_features = np.array([rhd_utils.extract_time_domain_features(window) for window in windows])

    # Plot each time-domain feature
    time_axis = np.arange(0, len(windows) * (window_size - overlap), window_size - overlap) / sample_rate

    fig, ax = plt.subplots(6, 1, figsize=(10, 12))
    feature_names = ['IEMG', 'MAV', 'SSI', 'RMS', 'VAR', 'MYOP']

    for i in range(6):
        ax[i].plot(time_axis, time_features[:, i])
        ax[i].set_title(feature_names[i])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def plot_wavelet_features(emg_signal, wavelet='db1', sample_rate=4000, window_size=400, overlap=200):
    """
    Plots wavelet features of EMG signals.

    Args:
        emg_signal: Raw EMG signal (1D array).
        wavelet: Type of wavelet to use for wavelet transform.
        sample_rate: Sampling rate of the EMG signal (default: 4000 Hz).
        window_size: Number of samples per window (default: 400).
        overlap: Number of overlapping samples (default: 200).

    Example usage with a sample EMG signal:
        plot_wavelet_features(emg_signal)

    """
    # Preprocess EMG signal (windowing)
    windows = rhd_utils.window_data(emg_signal, window_size=window_size, overlap=overlap)

    # Plot the wavelet decomposition coefficients for the first window as an example
    example_window = windows[0]
    coeffs = pywt.wavedec(example_window, wavelet, level=2)

    fig, ax = plt.subplots(len(coeffs), 1, figsize=(10, 8))

    for i, coeff in enumerate(coeffs):
        ax[i].plot(coeff)
        ax[i].set_title(f'Wavelet Coefficients Level {i + 1}')

    plt.tight_layout()
    plt.show()


def plot_feature_correlation(emg_signal, sample_rate=4000, window_size=400, overlap=200):
    """
    Plots a heatmap showing the correlation between different EMG features.

    Args:
        emg_signal: Raw EMG signal (1D array).
        sample_rate: Sampling rate of the EMG signal (default: 4000 Hz).
        window_size: Number of samples per window (default: 400).
        overlap: Number of overlapping samples (default: 200).
        
    Example usage with a sample EMG signal:
        plot_feature_correlation(emg_signal)

    """
    # Preprocess EMG signal (windowing)
    windows = rhd_utils.window_data(emg_signal, window_size=window_size, overlap=overlap)

    # Extract features for each window
    feature_matrix = np.array([rhd_utils.extract_features_from_window(window) for window in windows])

    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(feature_matrix.T)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()



