#!/usr/bin/env python3
"""
Compare EMG data from saved RHD file vs live TCP stream.

Useful for validating that TCP streaming produces identical data to file recording.
Plots both signals overlaid to check for alignment and amplitude matching.

Usage:
    python compare_rhd_vs_stream.py --rhd_file recording.rhd --channel 15 --samples 8000
"""
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from intan.io import load_rhd_file
from intan.interface import IntanRHXDevice


def main():
    """Main entry point for RHD vs stream comparison."""
    ap = argparse.ArgumentParser(
        description="Compare RHD file data with live TCP stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--rhd_file", required=True, help="Path to .rhd file")
    ap.add_argument("--channel", type=int, default=15, help="Channel index to compare")
    ap.add_argument("--samples", type=int, default=8000, help="Number of samples to compare")
    ap.add_argument("--duration", type=float, default=2.0, help="TCP recording duration (s)")
    args = ap.parse_args()

    # ================= TCP Section ==================
    print(f"[TCP] Connecting to RHX device...")
    try:
        with IntanRHXDevice() as device:
            if not device.connected:
                print("[ERROR] Failed to connect to RHX device.")
                print("Ensure RHX software is running with TCP server enabled.")
                return 1

            print(f"[TCP] Enabling channel {args.channel}...")
            device.enable_wide_channel(args.channel)

            print(f"[TCP] Recording {args.duration}s of data...")
            emg_data = device.record(duration_sec=args.duration)  # shape: (channels, samples)
            
            if emg_data is None or emg_data.shape[1] == 0:
                print("[ERROR] No data received from TCP stream.")
                return 1
            
            # Extract single channel and create timestamps
            voltages = emg_data[0]  # First (and only) enabled channel
            fs_tcp = float(device.sample_rate)
            timestamps = np.array([i / fs_tcp for i in range(len(voltages))])
            print(f"[TCP] Received {len(voltages)} samples at {fs_tcp:.1f} Hz")

    except Exception as e:
        print(f"[ERROR] TCP streaming failed: {e}")
        return 1

    # ============== .rhd File Section ==============
    print(f"[RHD] Loading file: {args.rhd_file}...")
    try:
        rhd_result = load_rhd_file(args.rhd_file)
    except Exception as e:
        print(f"[ERROR] Failed to load RHD file: {e}")
        return 1

    print(f"[RHD] Data shape: {rhd_result['amplifier_data'].shape}")

    # Extract comparison data
    rhd_data = rhd_result["amplifier_data"][:, :args.samples]
    fs_rhd = rhd_result["frequency_parameters"]["amplifier_sample_rate"]
    ts_rhd = rhd_result["t_amplifier"][:args.samples]
    print(f"[RHD] Sampling rate: {fs_rhd:.1f} Hz")

    # ================= Comparison Section ==================
    print(f"[PLOT] Comparing channel {args.channel}...")
    
    # Align lengths (use minimum)
    n_compare = min(len(voltages), len(ts_rhd), args.samples)
    
    # Plot both signals
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ts_rhd[:n_compare], rhd_data[args.channel, :n_compare], label="RHD File", linewidth=1, alpha=0.8)
    plt.plot(timestamps[:n_compare], voltages[:n_compare], '--', label="Live TCP", linewidth=1, alpha=0.8)
    plt.title(f"Channel {args.channel} Comparison — First {n_compare} Samples")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Difference plot
    plt.subplot(2, 1, 2)
    diff = rhd_data[args.channel, :n_compare] - voltages[:n_compare]
    plt.plot(ts_rhd[:n_compare], diff, color='red', linewidth=1)
    plt.title("Difference (RHD - TCP)")
    plt.xlabel("Time (s)")
    plt.ylabel("Difference (µV)")
    plt.grid(True, alpha=0.3)
    
    # Statistics
    print(f"\n[STATS] Comparison statistics:")
    print(f"  Mean difference: {diff.mean():.3f} µV")
    print(f"  Std difference:  {diff.std():.3f} µV")
    print(f"  Max difference:  {abs(diff).max():.3f} µV")
    
    plt.tight_layout()
    plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

