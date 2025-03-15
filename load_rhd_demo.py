"""
Quick demo that shows how to load an .rhd file recorded from the Intan RHX controller and visualize some of the data
"""

import utilities.rhd_utilities as rhd_utils
import utilities.plotting_utilities as plot_utils

if __name__ == "__main__":

    # ========== Load the data ==========
    # result, _ = rhd_utils.load_file('C:/absolute/path/to/emg/file.rhd') # Specify the file path...
    result, _ = rhd_utils.load_file()  # ...or use the file dialog to select the file

    # === If we have multiple files (for example, Intan saves separate files in 60 second increments) we can load and concatenate them ===
    #result, _ = rhd_utils.load_files_from_path(concatenate=True) # Specify folder or use file dialog

    # Get some data from the file to help
    emg_data = result.get('amplifier_data')                                   # Shape: (num_channels, num_samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']              # Sampling frequency
    t_s = result.get('t_amplifier')                                           # t_amplifier contains the time vector
    analog_data = result.get('board_adc_data')                                # Shape: (num_channels, num_samples)

    # ==== Display names of all available channels ====
    rhd_utils.print_all_channel_names(result)

    # ==== For multi-channel visualization, we can do a waterfall plot ====
    plot_utils.waterfall_plot(emg_data, range(128), t_s, plot_title='Intan EMG data')

    # ==== For single channel visualization ====
    ch_name = result.get('amplifier_channels')[4].get('native_channel_name')  # Get the name of the 5th channel "A-005"
    rhd_utils.plot_channel_by_name(ch_name, result)  # By name
    rhd_utils.plot_channel_by_index(8, result)  # By index

    # ====== Plot the data from analog_data ====
    if analog_data is not None:
        plot_utils.plot_figure(analog_data[0, :], t_s, 'Analog data')
