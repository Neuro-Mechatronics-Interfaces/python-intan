#!/usr/bin/env python3
"""
Scrolling live EMG waveform display using PyQtGraph.

Real-time visualization of multiple EMG channels with smooth scrolling display.
More efficient than matplotlib for continuous streaming applications.

Usage:
    python scrolling_live.py
    
Keyboard:
    Ctrl+C or close window to exit
"""
import sys
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from intan.interface import IntanRHXDevice


class ScrollingEMGPlot:
    """Real-time scrolling EMG plot with PyQtGraph."""
    
    def __init__(self, channels=[10, 11, 12, 13], window_sec=4, refresh_ms=200, y_range=(-200, 200)):
        # === Device Setup ===
        print("[INIT] Connecting to RHX device...")
        self.device = IntanRHXDevice()
        if not self.device.connected:
            raise RuntimeError(
                "Failed to connect to Intan device.\n"
                "Ensure RHX software is running with TCP server enabled."
            )

        self.device.enable_wide_channel(channels)
        self.device.set_blocks_per_write(1)

        self.channels = channels
        self.n_channels = len(channels)
        self.sampling_rate = float(self.device.sample_rate)
        self.refresh_ms = refresh_ms
        self.window_sec = window_sec
        self.window_ms = int(refresh_ms)  # Request window in ms
        self.buffer_size = int(window_sec * self.sampling_rate)
        
        print(f"[OK] Connected. Sample rate: {self.sampling_rate:.1f} Hz")
        print(f"[PLOT] Channels: {channels}, Window: {window_sec}s")

        # === Plot Buffers ===
        self.buffers = {
            ch: deque([0.0] * self.buffer_size, maxlen=self.buffer_size)
            for ch in self.channels
        }

        # === Qt App and Window ===
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="Scrolling EMG Plot (Intan)")
        self.win.setWindowTitle("Live EMG Viewer")

        self.curves = {}
        self.plots = {}
        self.x = np.linspace(-self.window_sec, 0, self.buffer_size)

        for ch in self.channels:
            plot = self.win.addPlot(title=f"Channel A-{ch:03d}")
            plot.setLabel("left", "μV")
            plot.setLabel("bottom", "Time (s)")
            plot.setYRange(*y_range)
            curve = plot.plot(pen=pg.mkPen("cyan", width=1))
            self.curves[ch] = curve
            self.plots[ch] = plot
            self.win.nextRow()

        self.win.show()

        # === Update Timer ===
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.refresh_ms)

        # Graceful shutdown
        self.app.aboutToQuit.connect(self.cleanup)

    def update(self):
        """Update plot with new data from device."""
        try:
            # Use get_latest_window instead of deprecated stream method
            emg_data = self.device.get_latest_window(window_ms=self.window_ms)
            if emg_data is None or emg_data.shape[1] == 0:
                return
            
            for i, ch in enumerate(self.channels):
                if i < emg_data.shape[0]:
                    self.buffers[ch].extend(emg_data[i])
                    self.curves[ch].setData(self.x, list(self.buffers[ch]))
        except Exception as e:
            print(f"[ERROR] Update failed: {e}")
            self.cleanup()

    def cleanup(self):
        """Clean shutdown of device connection."""
        print("[STOP] Shutting down...")
        try:
            self.device.stop_streaming()
            self.device.close()
        except Exception:
            pass
        print("[DONE] Connection closed.")

    def run(self):
        """Start the Qt event loop."""
        self.device.start_streaming()
        print("[RUN] Streaming... Close window or press Ctrl+C to exit.")
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    try:
        plotter = ScrollingEMGPlot(
            channels=[10, 11, 12, 13],
            window_sec=4,
            refresh_ms=200,
            y_range=(-200, 200)
        )
        plotter.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
