#!/usr/bin/env python3
"""
Record EMG data from Intan RHX device and display a sample channel.

Demonstrates basic device connection, channel configuration, data recording,
and visualization using matplotlib.

Usage:
    python record_demo.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from intan.interface import IntanRHXDevice


def main():
    """Main recording demonstration."""
    DURATION_SEC = 10
    NUM_CHANNELS = 128
    DISPLAY_CHANNEL = 5

    print(f"[INIT] Connecting to RHX device...")
    
    # Use context manager for proper cleanup
    try:
        with IntanRHXDevice(num_channels=NUM_CHANNELS) as device:
            if not device.connected:
                print("[ERROR] Failed to connect to RHX device.")
                print("Ensure RHX software is running with TCP server enabled.")
                return 1

            SAMPLE_RATE = float(device.sample_rate)
            print(f"[OK] Connected. Sample rate: {SAMPLE_RATE:.1f} Hz")

            # Configure channels
            print(f"[SETUP] Enabling {NUM_CHANNELS} channels...")
            device.enable_wide_channel(range(NUM_CHANNELS))
            device.set_blocks_per_write(8)

            # Record data
            print(f"[RECORD] Recording {DURATION_SEC}s of data...")
            try:
                emg = device.record(duration_sec=DURATION_SEC)
            except KeyboardInterrupt:
                print("\n[STOP] Recording interrupted by user.")
                return 1

            if emg is None or emg.shape[1] == 0:
                print("[ERROR] No data recorded.")
                return 1

            print(f"[OK] Recorded {emg.shape[1]} samples across {emg.shape[0]} channels")

            # Plot sample channel
            t = np.arange(emg.shape[1]) / SAMPLE_RATE
            
            plt.figure(figsize=(12, 4))
            plt.plot(t, emg[DISPLAY_CHANNEL])
            plt.title(f"EMG Signal - Channel {DISPLAY_CHANNEL}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (µV)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print("[DONE] Recording complete.")
            return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
