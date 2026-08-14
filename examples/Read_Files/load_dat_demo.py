"""Load an Intan per-signal ``.dat`` recording directory and optionally save EMG as NPZ."""

import argparse
from pathlib import Path

import numpy as np

from intan.io import load_dat_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        help="Recording directory. Omit to open the package file picker.",
    )
    parser.add_argument("--save-npz", action="store_true", help="Save downsampled EMG beside the recording")
    parser.add_argument("--target-fs", type=float, default=4000.0, help="Target sampling rate in Hz")
    args = parser.parse_args()

    result = load_dat_file(args.directory) if args.directory else load_dat_file()
    emg_data = result["amplifier_data"]
    fs = float(result["frequency_parameters"]["amplifier_sample_rate"])
    t_s = result["t_amplifier"]
    print(f"EMG data shape: {emg_data.shape}")
    print(f"Sampling frequency: {fs:g} Hz")

    if args.target_fs <= 0 or args.target_fs > fs:
        parser.error("--target-fs must be positive and no greater than the source sampling rate")
    factor = max(1, round(fs / args.target_fs))
    emg_data = emg_data[:, ::factor]
    t_s = t_s[::factor]
    output_fs = fs / factor
    print(f"Downsampled to {output_fs:g} Hz")

    if args.save_npz:
        base = Path(result["export_basepath"])
        label = result.get("export_basename") or base.name or "recording"
        output = base / f"{label}_emg_data.npz"
        np.savez_compressed(output, emg_data=emg_data, t_s=t_s, fs=output_fs)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
