#!/usr/bin/env python3
"""
Real-time multichannel waveform visualization using matplotlib animation.

Displays multiple EMG channels in stacked subplots with scrolling time window.
Uses matplotlib's blitting for efficient updates.

Usage:
    python multichannel_stream_plot.py
    
Configuration (edit script):
    CHANNELS: List of channel indices to display
    WINDOW_SEC: Display window duration in seconds
    PLOT_UPDATE_HZ: Refresh rate (lower = less CPU)
"""
import time
import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # Ensure fast, responsive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from intan.interface import IntanRHXDevice

# === Configuration ===
CHANNELS = [10, 11, 12, 13, 14, 15]
WINDOW_SEC = 1.0
PLOT_UPDATE_HZ = 5  # Refresh every 200 ms

PLOT_INTERVAL_SEC = 1 / PLOT_UPDATE_HZ

# === Initialize RHX Device ===
print("[INIT] Connecting to RHX device...")
device = IntanRHXDevice()
if not device.connected:
    print("[ERROR] Could not connect to RHX TCP server.")
    print("Ensure RHX software is running with TCP server enabled.")
    sys.exit(1)

device.enable_wide_channel(CHANNELS)
device.set_blocks_per_write(1)

# Get actual sampling rate from device
SAMPLING_RATE = float(device.sample_rate)
print(f"[OK] Connected. Sampling rate: {SAMPLING_RATE:.1f} Hz")

SAMPLES_PER_WINDOW = int(SAMPLING_RATE * WINDOW_SEC)
x_data = np.linspace(-WINDOW_SEC, 0, SAMPLES_PER_WINDOW)

# === Setup Plotting ===
fig, axs = plt.subplots(len(CHANNELS), 1, figsize=(10, 6), sharex=True)
axs = np.atleast_1d(axs)
lines = {}
buffers = {}

for ax, ch in zip(axs, CHANNELS):
    ax.set_ylabel("µV")
    ax.set_ylim(-200, 200)
    ax.grid(False)
    ax.set_title("")
    ax.legend().set_visible(False)

    line, = ax.plot(x_data, np.zeros(SAMPLES_PER_WINDOW), label=f"A-{ch:03d}", color="cyan", linewidth=1)
    lines[ch] = line
    buffers[ch] = np.zeros(SAMPLES_PER_WINDOW)

axs[-1].set_xlabel("Time (s)")
fig.suptitle("Live EMG: " + ", ".join([f"A-{ch:03d}" for ch in CHANNELS]))
fig.tight_layout(rect=[0, 0, 1, 0.96])

# === Update Function ===
def update_plot(_):
    """Update plot with new streaming data."""
    window_ms = int(PLOT_INTERVAL_SEC * 1000)
    start_time = time.time()
    try:
        # Use get_latest_window instead of deprecated stream method
        channel_array = device.get_latest_window(window_ms=window_ms)
        if channel_array is None or channel_array.shape[1] == 0:
            return lines.values()
    except Exception as e:
        print(f"[ERROR] Stream failed: {e}")
        return lines.values()

    elapsed = time.time() - start_time
    for i, ch in enumerate(CHANNELS):
        new_data = channel_array[i]
        num_new = min(len(new_data), SAMPLES_PER_WINDOW)
        buffers[ch] = np.roll(buffers[ch], -num_new)
        buffers[ch][-num_new:] = new_data[-num_new:]
        lines[ch].set_ydata(buffers[ch])

    rate = channel_array.shape[1] / elapsed if elapsed > 0 else 0
    print(f"{channel_array.shape[1]} samples/ch in {elapsed:.3f} s → {rate:.1f} Hz")
    return lines.values()

# === Animation Loop ===
ani = animation.FuncAnimation(
    fig, update_plot,
    interval=int(PLOT_INTERVAL_SEC * 1000),
    blit=True
)

try:
    print(f"[RUN] Streaming channels: {', '.join([f'A-{ch:03d}' for ch in CHANNELS])}")
    print("[RUN] Press Ctrl+C or close window to stop.")
    device.start_streaming()  # Ensure streaming is started
    plt.show()
except KeyboardInterrupt:
    print("\n[STOP] Interrupted.")
finally:
    try:
        device.stop_streaming()
        device.close()
    except Exception:
        pass
    print("[DONE] Connection closed.")
