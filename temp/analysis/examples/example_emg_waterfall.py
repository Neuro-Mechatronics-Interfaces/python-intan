"""
Quick demo showing the lading of RHD data and displaying the data from a user-specified channel name A-007

"""
import os
import sys

# Get the absolute path of the root project directory
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

import argparse
from intan_utilities import rhd_utilities as rhd_utils
from intan_utilities import plotting_utilities as plot_utils
from intan_utilities import emg_processing as emg_utils
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Specify the filename
    filename = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\Jonathan\raw\2024_11_11\index_241111_174705\index_241111_174705.rhd'

    parser = argparse.ArgumentParser(description='Load and plot RHD data')
    parser.add_argument('--filename', type=str, default=filename, help='Path to the RHD file')
    parser.add_argument('--channels', type=str, default='1-8', help='Range of channels to plot')
    parser.add_argument('--verbose', type=bool, default=False, help='Enable text debugging output')
    args = parser.parse_args()

    # Load the data
    result, data_present = rhd_utils.load_file(args.filename, verbose=args.verbose)

    if not data_present:
        print(f'No data present in {filename}')

    else:
        # parse the channel range so the string or single channel can be converted to a list of integers
        if '-' in args.channels:
            ch = args.channels.split('-')
            ch = list(range(int(ch[0])-1, int(ch[1])))
        else:
            ch = [int(args.channels)-1]
        print(f'Channels: {ch}')

        # Get the EMG channel data and time vector
        emg_data = result['amplifier_data'][ch, :]                    # Shape: (num_channels, num_samples)
        time_s = result['t_amplifier']                                # Time vector in seconds
        fs = result['frequency_parameters']['amplifier_sample_rate']  # Sampling frequency

        # Simple filtering of the data
        filtered_data = emg_utils.butter_bandpass_filter(emg_data.T, 20, 500, fs, 4)

        # Plot the data
        plot_utils.waterfall_plot_old(filtered_data.T, ch, time_s, plot_title='2024-10-22: Index Flexion')
        plt.show()
