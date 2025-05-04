""" Plotting functions for visualizing EMG data and features.

"""

#import pywt
#import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from intan.io._exceptions import ChannelNotFoundError
from intan.io._channel_utils import find_channel_in_header
#from matplotlib.collections import LineCollection



def plot_channel_by_name(channel_name, result):
    """Plots all data associated with channel specified as 'channel_name' in
    'result' dict.
    """
    # Find channel that corresponds to this name
    channel_found, signal_type, signal_index = find_channel_in_header(
        channel_name, result)

    # Plot this channel
    if channel_found:
        _, ax = plt.subplots()
        # fig, ax = plt.subplots()
        ax.set_title(channel_name)
        ax.set_xlabel('Time (s)')

        if signal_type == 'amplifier_channels':
            ylabel = 'Voltage (microVolts)'
            signal_data_name = 'amplifier_data'
            t_vector = result['t_amplifier']

        elif signal_type == 'aux_input_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'aux_input_data'
            t_vector = result['t_aux_input']

        elif signal_type == 'supply_voltage_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'supply_voltage_data'
            t_vector = result['t_supply_voltage']

        elif signal_type == 'board_adc_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'board_adc_data'
            t_vector = result['t_board_adc']

        elif signal_type == 'board_dig_in_channels':
            ylabel = 'Digital In Events (High or Low)'
            signal_data_name = 'board_dig_in_data'
            t_vector = result['t_dig']

        elif signal_type == 'board_dig_out_channels':
            ylabel = 'Digital Out Events (High or Low)'
            signal_data_name = 'board_dig_out_data'
            t_vector = result['t_dig']

        else:
            raise ChannelNotFoundError(
                'Plotting failed; signal type ', signal_type, ' not found')

        ax.set_ylabel(ylabel)
        ax.plot(t_vector, result[signal_data_name][signal_index, :])
        ax.margins(x=0, y=0)

        print("Plotting channel: ", channel_name)
        plt.show()

    else:
        raise ChannelNotFoundError(
            'Plotting failed; channel ', channel_name, ' not found')

def plot_channel_by_index(channel_index, result):
    """Plots all data associated with channel specified as 'channel_index' in
    'result' dict.
    """
    N_channels = len(result['amplifier_channels'])
    if channel_index >= N_channels:
        raise ChannelNotFoundError(
            'Plotting failed; channel index ', channel_index, ' not found')
    else:
        _, ax = plt.subplots()
        ax.set_title(result['amplifier_channels'][channel_index]['custom_channel_name'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (microVolts)')
        ax.plot(result['t_amplifier'], result['amplifier_data'][channel_index, :])
        ax.margins(x=0, y=0)

        print("Plotting channel: ", result['amplifier_channels'][channel_index]['custom_channel_name'])
        plt.show()


# def plot_time_domain_features(emg_signal, sample_rate=4000, window_size=400, overlap=200):
#     """
#     Plots time-domain features of EMG signals over time.
#
#     Args:
#         emg_signal: Raw EMG signal (1D array).
#         sample_rate: Sampling rate of the EMG signal (default: 4000 Hz).
#         window_size: Number of samples per window (default: 400).
#         overlap: Number of overlapping samples (default: 200).
#     """
#     # Preprocess EMG signal (windowing)
#     windows = rhd_utils.window_data(emg_signal, window_size=window_size, overlap=overlap)
#
#     # Extract time-domain features for each window
#     time_features = np.array([rhd_utils.extract_time_domain_features(window) for window in windows])
#
#     # Plot each time-domain feature
#     time_axis = np.arange(0, len(windows) * (window_size - overlap), window_size - overlap) / sample_rate
#
#     fig, ax = plt.subplots(6, 1, figsize=(10, 12))
#     feature_names = ['IEMG', 'MAV', 'SSI', 'RMS', 'VAR', 'MYOP']
#
#     for i in range(6):
#         ax[i].plot(time_axis, time_features[:, i])
#         ax[i].set_title(feature_names[i])
#         ax[i].set_xlabel('Time (s)')
#         ax[i].set_ylabel('Amplitude')
#
#     plt.tight_layout()
#     plt.show()


# def plot_wavelet_features(emg_signal, wavelet='db1', sample_rate=4000, window_size=400, overlap=200):
#     """
#     Plots wavelet features of EMG signals.
#
#     Args:
#         emg_signal: Raw EMG signal (1D array).
#         wavelet: Type of wavelet to use for wavelet transform.
#         sample_rate: Sampling rate of the EMG signal (default: 4000 Hz).
#         window_size: Number of samples per window (default: 400).
#         overlap: Number of overlapping samples (default: 200).
#
#     Example usage with a sample EMG signal:
#         plot_wavelet_features(emg_signal)
#
#     """
#     # Preprocess EMG signal (windowing)
#     windows = rhd_utils.window_data(emg_signal, window_size=window_size, overlap=overlap)
#
#     # Plot the wavelet decomposition coefficients for the first window as an example
#     example_window = windows[0]
#     coeffs = pywt.wavedec(example_window, wavelet, level=2)
#
#     fig, ax = plt.subplots(len(coeffs), 1, figsize=(10, 8))
#
#     for i, coeff in enumerate(coeffs):
#         ax[i].plot(coeff)
#         ax[i].set_title(f'Wavelet Coefficients Level {i + 1}')
#
#     plt.tight_layout()
#     plt.show()


# def plot_feature_correlation(emg_signal, sample_rate=4000, window_size=400, overlap=200):
#     """
#     Plots a heatmap showing the correlation between different EMG features.
#
#     Args:
#         emg_signal: Raw EMG signal (1D array).
#         sample_rate: Sampling rate of the EMG signal (default: 4000 Hz).
#         window_size: Number of samples per window (default: 400).
#         overlap: Number of overlapping samples (default: 200).
#
#     Example usage with a sample EMG signal:
#         plot_feature_correlation(emg_signal)
#
#     """
#     # Preprocess EMG signal (windowing)
#     windows = rhd_utils.window_data(emg_signal, window_size=window_size, overlap=overlap)
#
#     # Extract features for each window
#     feature_matrix = np.array([rhd_utils.extract_features_from_window(window) for window in windows])
#
#     # Compute the correlation matrix
#     correlation_matrix = np.corrcoef(feature_matrix.T)
#
#     # Plot heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#     plt.title('Feature Correlation Heatmap')
#     plt.show()

def plot_figure(y, x, title='Example Plot', x_label='time (s)', y_label='Y-axis', legend=True, fig_size=(10, 6)):
    """
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the data
    ax.plot(x, y, label='Data')

    # Add a title and labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add a legend
    if legend:
        ax.legend()

    # Show the plot
    plt.show()

def _add_scalebars(ax, scale_time=0.1, scale_amp=10):
    ax.annotate('', xy=(0, scale_amp), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-', lw=2))
    ax.annotate('', xy=(scale_time, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-', lw=2))

def _insert_channel_labels(ax, channel_labels, y_offsets):
    for label, y in zip(channel_labels, y_offsets):
        ax.text(-0.05, y, label, va='center', ha='right')

def _insert_vertical_labels(ax, labels, x_positions):
    for label, x in zip(labels, x_positions):
        ax.axvline(x, color='gray', linestyle='--')
        ax.text(x, ax.get_ylim()[1], label, rotation=90, va='bottom', ha='center')

def plot_figure(y, x, title='Example Plot', x_label='time (s)', y_label='Y-axis', legend=True, fig_size=(10, 6)):
    """
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the data
    ax.plot(x, y, label='Data')

    # Add a title and labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add a legend
    if legend:
        ax.legend()

    # Show the plot
    plt.show()


