"""
Quick demo that shows how to load an .rhd file recorded from the Intan RHX controller and visualize some of the data
"""

from intan.io import load_rhd_file, print_all_channel_names
from intan.plotting import waterfall, plot_channel_by_index, plot_channel_by_name, plot_figure

if __name__ == "__main__":

    # ========== Load the data ==========
    #file_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\Jonathan\raw\2024_10_22\IsometricFlexionRamp_1_241022_150834\IsometricFlexionRamp_1_241022_150834.rhd'
    #file_path = r'C:\Users\HP\Desktop\Temp\HDEMG_comparison_042325\rest_250423_164915\rest_250423_164915.rhd'
    #file_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\Jonathan\raw\2024_10_22\WristFlexion_5x_241022_142239\WristFlexion_5x_241022_142239.rhd'
    file_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\raw\2024_11_11\index2_241111_183222\index2_241111_183222.rhd'
    #file_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\Jonathan\raw\2024_10_22\IndexFlexion_5x_241022_143429\IndexFlexion_5x_241022_143429.rhd'
    result = load_rhd_file(file_path)  # Load a specific file by path
    # result, _ = rhd_utils.load_file('C:/absolute/path/to/emg/file.rhd') # Specify the file path...
    #result, _ = rhd_utils.load_file()  # ...or use the file dialog to select the file

    # === If we have multiple files (for example, Intan saves separate files in 60 second increments) we can load and concatenate them ===
    #result, _ = rhd_utils.load_files_from_path(concatenate=True) # Specify folder or use file dialog

    # Get some data from the file to help
    emg_data = result.get('amplifier_data')                                   # Shape: (num_channels, num_samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']              # Sampling frequency
    t_s = result.get('t_amplifier')                                           # t_amplifier contains the time vector
    analog_data = result.get('board_adc_data')                                # Shape: (num_channels, num_samples)
    print(result.keys())

    # ==== Display names of all available channels ====
    print_all_channel_names(result)

    # ==== For multi-channel visualization, we can do a waterfall plot ====
    waterfall(emg_data, range(64, 128), t_s, plot_title='Intan EMG data')

    # ==== For single channel visualization ====
    ch_name = result.get('amplifier_channels')[4].get('native_channel_name')  # Get the name of the 5th channel "A-005"
    plot_channel_by_name(ch_name, result)  # By name
    plot_channel_by_index(8, result)  # By index

    # Flip the analog data
    analog_data = -1 * analog_data if analog_data is not None else None  # Example to flip the analog data if needed

    # ====== Plot the data from analog_data ====
    if analog_data is not None:
        plot_figure(analog_data[0, :], t_s, 'Analog data')