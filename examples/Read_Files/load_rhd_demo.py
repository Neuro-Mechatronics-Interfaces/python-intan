"""
Quick demo that shows how to load an .rhd file recorded from the Intan RHX controller and visualize some of the data
"""
import os
from intan.io import load_rhd_file, print_all_channel_names, save_as_npz
from intan.plotting import waterfall, plot_channel_by_index, plot_channel_by_name, plot_figure

if __name__ == "__main__":

    save = True

    # ========== Load the data ==========
    # result = load_rhd_file('C:/absolute/path/to/emg/file.rhd') # Specify the file path...
    # result = load_rhd_file(r"G:\Shared drives\NML_shared\DataShare\HDEMG_SCI\MCP01_NML-EMG\2024_12_10\raw\MVC_241210_172457\MVC_241210_172457.rhd")
    result = load_rhd_file()                                   # ...or use the file dialog to select the file

    # === If we have multiple files (for example, Intan saves separate files in 60 second increments) we can load and concatenate them ===
    #result = load_rhd_file(merge_files=True)                     # ...use the file dialog to select all files

    # Get some data from the file to help
    emg_data = result.get('amplifier_data')                                   # Shape: (num_channels, num_samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']              # Sampling frequency
    t_s = result.get('t_amplifier')                                           # t_amplifier contains the time vector
    analog_data = result.get('board_adc_data')                                # Shape: (num_channels, num_samples)
    print(result.keys())

    # ==== Display names of all available channels ====
    print(result['channel_names'])

    # ==== For multi-chqannel visualization, we can do a waterfall plot ====
    #waterfall(None, emg_data, range(0, 127), t_s, plot_title='Intan EMG data')

    # ==== For single channel visualization ====
    #ch_name = result.get('amplifier_channels')[4].get('native_channel_name')  # Get the name of the 5th channel "A-005"
    #plot_channel_by_name(ch_name, result)  # By name
    #plot_channel_by_index(8, result)  # By index

    # ====== Plot the data from analog_data ====
    #if analog_data is not None:
    #    plot_figure(analog_data[0, :], t_s, 'Analog data')

    label = result['export_basename']
    if save:
        save_as_npz(result, os.path.join(result['export_basepath'], f"{label}.npz"))
