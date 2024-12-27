"""
Quick demo showing the lading of RHD data and displaying the data from a user-specified channel name A-007

"""
import utilities.rhd_utilities as rhd_utils
from utilities import plotting_utilities as plot_utils
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Specify the filename
    #filename = 'path/to/emg/file.rhd'
    filename = filename = '/mnt/c/Users/NML/Desktop/hdemg_test/MCP01/2024_11_12/wrist_flexion_241112_170600/wrist_flexion_241112_170600.rhd'

    # Load the data
    result, data_present = rhd_utils.load_file(filename)

    # Display names of all available channels
    rhd_utils.print_all_channel_names(result)

    if data_present:

        # For single channel visualization
        #channel_name = 'A-007'  # Change this variable and re-run cell to plot a different channel
        #rhd_utils.plot_channel(channel_name, result)

        # For multi-channel visualization
        emg_data = result['amplifier_data']  # Shape: (num_channels, num_samples)
        channels_to_plot = range(128)  # Channels 000 to 127
        time_vector = result['t_amplifier']  # Assuming t_amplifier contains the time vector

        plot_utils.waterfall_plot_old(emg_data, channels_to_plot, time_vector,
                                  plot_title='2024-10-22: Index Flexion'
                                  )
        plt.show()
    else:
        print('Plotting not possible; no data in this file')



