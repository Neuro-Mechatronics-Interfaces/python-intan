"""
Demo script showing how to load a .dat file recorded from the Intan RHX controller and visualize some of the data.
"""
import os
import utilities.plotting_utilities as plot_utils
from utilities.intan_utilities import load_dat_file

if __name__ == "__main__":

    # Define the file path to the .rhd header file
    root_dir = r'G:\Shared drives\NML_shared\DataShare\HDEMG_SCI\MCP01_NML-EMG\2025_03_21\raw\MVCRest_250321_165434'

    results = load_dat_file(root_dir)
    print(results.keys())
    emg_data = results['amplifier_data']  # Shape: (num_channels, num_samples)
    t_s = results['t_amplifier']  # Assuming t_amplifier contains the time vector
    #plot_utils.waterfall_plot(emg_data, range(0, 128), t_s, plot_title='Intan EMG data')
