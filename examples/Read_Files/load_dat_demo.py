"""
Demo script showing how to load a .dat file recorded from the Intan RHX controller and visualize some of the data.
"""
import os
from intan.io import load_dat_file,  save_as_npz

if __name__ == "__main__":

    save = True

    # ========== Load the data ==========
    # result = load_dat_file('C:/absolute/path/to/emg/file.dat') # Specify the file path...
    # result = load_dat_file()                                   # ...or use the file dialog to select the file

    # Define the file path to the .rhd header file
    root_dir = r'G:\Shared drives\NML_shared\DataShare\HDEMG_SCI\MCP01_NML-EMG\2025_03_21\raw\AllData_250321_171530'
    result = load_dat_file(root_dir)

    print(result.keys())
    emg_data = result['amplifier_data']  # Shape: (num_channels, num_samples)
    t_s = result['t_amplifier']  # Assuming t_amplifier contains the time vector

    #label = result['export_basename']
    label = "AllData_250321_171530"
    if save:
        save_as_npz(result, os.path.join(result['export_basepath'], f"{label}.npz"))
