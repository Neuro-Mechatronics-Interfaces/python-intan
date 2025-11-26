import os
import argparse

from intan.io import load_csv_file, save_as_npz  # already in your package
from intan.plotting import waterfall

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load and visualize Intan .rhd files")
    parser.add_argument('--save_npz', action='store_true', help='Save the loaded data as an NPZ file')
    args = parser.parse_args()

    # Load your CSV (will also work with a file dialog if you omit the path)
    #result = load_csv_file(r"/path/to/emg_imu_data.csv")
    result = load_csv_file()  # use file dialog

    # Result structure highlights
    emg = result["amplifier_data"]            # shape: (n_channels, n_samples)
    t   = result["t_amplifier"]               # shape: (n_samples,)
    imu = result.get("board_adc_data", None)  # optional: (n_imu_channels, n_samples)
    print(f"Loaded EMG data shape: {emg.shape}, time vector shape: {t.shape}")
    if imu is not None:
        print(f"Loaded IMU data shape: {imu.shape}")

    # Apply CAR referencing across EMG channels
    emg = emg - emg.mean(axis=0)  # Common Average Referencing

    waterfall(None, emg, range(0,emg.shape[0]), t,
              offset_increment=400,
              plot_title='EMG data',
              downsampling_factor=10,
              )

    # Save to NPZ in your usual flow
    if args.save_npz:
        save_as_npz(result, os.path.join(result["export_basepath"], f"{result['export_basename']}.npz"))
