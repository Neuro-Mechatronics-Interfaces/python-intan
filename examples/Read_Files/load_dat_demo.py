"""
Demo script showing how to load a .dat file recorded from the Intan RHX controller and visualize some of the data.
"""
import os
from intan.io import load_dat_file, save_as_npz

if __name__ == "__main__":

    save = True

    # ========== Load the data ==========
    # result = load_dat_file('C:/absolute/path/to/emg/file.dat') # Specify the file path...
    # result = load_dat_file()                                   # ...or use the file dialog to select the file

    # Define the file path to the .rhd header file
    #root_dir = r'G:\Shared drives\NML_shared\DataShare\HDEMG_SCI\MCP01_NML-EMG\2025_03_21\raw\AllData_250321_171530'
    root_dir = r"G:\Shared drives\NML_shared\DataShare\HDEMG_SCI\MCP01_NML-EMG\2025_03_21\raw\AllData_250321_165927"
    result = load_dat_file(root_dir)

    print(result.keys())
    emg_data = result['amplifier_data']  # Shape: (num_channels, num_samples)
    # print the shape of teh data
    print("EMG data shape:", emg_data.shape)
    fs = result['frequency_parameters']['amplifier_sample_rate']  # Sampling frequency
    print(f"Sampling frequency: {fs} Hz")
    t_s = result['t_amplifier']  # Assuming t_amplifier contains the time vector

    # assuming the data is collected at 20,000Hz, downsample time and data to 4kHz
    downsample_factor = int(fs / 4000)
    emg_data = emg_data[:, ::downsample_factor]
    t_s = t_s[::downsample_factor]
    fs = 4000.0
    print(f"Downsampled to {fs} Hz")

    #label = result['export_basename']
    #label = "AllData_250321_171530"
    label = "AllData_250321_165927"

    if save:
        # just save the emg data and timestamps as npz
        import numpy as np
        np.savez_compressed(os.path.join(result['export_basepath'], f"{label}_emg_data.npz"),
                            emg_data=emg_data,
                            t_s=t_s,
                            fs=fs)
        #save_as_npz(result, os.path.join(result['export_basepath'], f"{label}.npz"))
