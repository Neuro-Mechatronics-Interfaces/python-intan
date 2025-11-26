"""
Quick demo that shows how to load an .rhd file recorded from the Intan RHX controller and visualize some of the data
"""
import os
import argparse
from intan.io import load_rhd_file, save_as_npz
from intan.plotting import waterfall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and visualize Intan .rhd files")
    parser.add_argument('--save_npz', action='store_true', help='Save the loaded data as an NPZ file')
    args = parser.parse_args()

    # ========== Load the data ==========
    # result = load_rhd_file(r'C:/absolute/path/to/emg/file.rhd')             # Specify the file path...
    result = load_rhd_file()                                                  # ...or use the file dialog to select the file

    # === If we have multiple files (for example, Intan saves separate files in 60 second increments) we can load and concatenate them ===
    #result = load_rhd_file(merge_files=True)                                 # ...use the file dialog to select all files

    # Get some data from the file to help
    emg = result.get('amplifier_data')                                        # Shape: (num_channels, num_samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']              # Sampling frequency
    t_s = result.get('t_amplifier')                                           # t_amplifier contains the time vector
    analog_data = result.get('board_adc_data')                                # Shape: (num_channels, num_samples)
    print(result.keys())

    # ==== Display names of all available channels ====
    print(result['channel_names'])

    # ==== For multi-chqannel visualization, we can do a waterfall plot ====
    waterfall(None, emg, range(0, 127), t_s, plot_title='Intan EMG data')

    if args.save_npz:
        save_as_npz(result, os.path.join(result['export_basepath'], f"{result['export_basename']}.npz"))
