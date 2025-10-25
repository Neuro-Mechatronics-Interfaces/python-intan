import sys
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from intan.interface import IntanRHXDevice


class ScrollingEMGPlot:
    def __init__(self, channels=[10, 11, 12, 13], window_sec=4, refresh_ms=200, y_range=(-200, 200)):
        # === Device Setup ===
        self.device = IntanRHXDevice()
        if not self.device.connected:
            raise RuntimeError("Failed to connect to Intan device. Is the TCP server running?")

        self.device.configure(channels=channels, blocks_per_write=1, enable_wide=True)
        self.device.flush_commands()

        self.channels = channels
        self.n_channels = len(channels)
        self.sampling_rate = self.device.sample_rate
        self.refresh_ms = refresh_ms
        self.window_sec = window_sec
        self.frames_per_update = int((refresh_ms / 1000) * self.sampling_rate)
        self.buffer_size = int(window_sec * self.sampling_rate)

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
        try:
            _, emg_data = self.device.stream(n_frames=self.frames_per_update)
            for i, ch in enumerate(self.channels):
                self.buffers[ch].extend(emg_data[i])
                self.curves[ch].setData(self.x, list(self.buffers[ch]))
        except Exception as e:
            print(f"[ERROR] Update failed: {e}")
            self.cleanup()

    def cleanup(self):
        print("Shutting down...")
        self.device.close()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    plotter = ScrollingEMGPlot(
        channels=[10, 11, 12, 13],
        window_sec=4,
        refresh_ms=200,
        y_range=(-200, 200)
    )
    plotter.run()
