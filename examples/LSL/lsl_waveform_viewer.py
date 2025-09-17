#!/usr/bin/env python3
# examples/lsl_waveform_viewer.py
"""
Live LSL waveform viewer (pyqtgraph).

Subscribes to a numeric LSL stream (type/name/source_id) and plots selected channels
in a rolling time window. Requires your package to export LSLSubscriber/LSLStreamSpec.

Usage:
  python -m examples.lsl_waveform_viewer --type EMG --channels 0,1,2 --win 2.0

Keyboard:
  Q / Esc  Quit
"""

import argparse
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# Import your subscriber
from intan.interface import LSLSubscriber, LSLStreamSpec


def parse_channels(s: str):
    """
    Parse "0,2,5-8" → [0,2,5,6,7,8]
    """
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    # unique, sorted
    return sorted(set(out))


class WaveViewer(QtWidgets.QWidget):
    def __init__(self, sub: LSLSubscriber, channels, win_s=2.0, downsample=1, ylabel="amplitude (au)"):
        super().__init__()
        self.sub = sub
        self.channels = list(channels)
        self.win_s = float(win_s)
        self.downsample = max(1, int(downsample))

        self.setWindowTitle("LSL Waveform Viewer")
        layout = QtWidgets.QVBoxLayout(self)

        # Plot setup
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")  # uses palette in dark UIs nicely; change to None for default
        self.plot.addLegend(offset=(10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("left", ylabel)
        self.plot.setLabel("bottom", "time (s)")
        layout.addWidget(self.plot, 1)

        # Curves for each channel
        self.curves = []
        for ch in self.channels:
            curve = self.plot.plot(pen=pg.mkPen(width=1.8), name=f"ch{ch}")
            self.curves.append(curve)

        # Timer for UI refresh (~60 Hz)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._update)
        self.timer.start(16)

        # Shortcuts
        QtWidgets.QShortcut(QtCore.Qt.Key_Escape, self, activated=self.close)
        QtWidgets.QShortcut(QtCore.Qt.Key_Q, self, activated=self.close)

    def _update(self):
        X = self.sub.read_window(self.win_s)  # (T, C)
        if X.size == 0:
            return
        # Downsample for rendering if needed
        if self.downsample > 1:
            X = X[::self.downsample, :]

        T = X.shape[0]
        t = np.linspace(-self.win_s, 0.0, T, dtype=np.float32)  # seconds, oldest→newest

        for curve, ch in zip(self.curves, self.channels):
            if ch < X.shape[1]:
                y = X[:, ch]
                curve.setData(t, y)


def main():
    ap = argparse.ArgumentParser(description="Subscribe to an LSL numeric stream and plot waveforms.")
    ap.add_argument("--type", default="EMG", help="LSL stream type (e.g., EMG)")
    ap.add_argument("--name", default=None, help="LSL stream name to prefer")
    ap.add_argument("--source_id", default=None, help="Exact source_id to pin to")
    ap.add_argument("--timeout", type=float, default=3.0, help="Resolve timeout (s)")
    ap.add_argument("--channels", default="0-3", help="Channels to show, e.g. '0,2,5-8'")
    ap.add_argument("--win", type=float, default=2.0, help="Window length in seconds")
    ap.add_argument("--downsample", type=int, default=1, help="Render every Nth sample")
    ap.add_argument("--ylabel", default="amplitude (au)", help="Y axis label")
    ap.add_argument("--verbose", action="store_true", help="Print stream resolution info")
    args = ap.parse_args()

    chs = parse_channels(args.channels)

    spec = LSLStreamSpec(
        name=args.name,
        stype=args.type,
        source_id=args.source_id,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    sub = LSLSubscriber(spec, fs_hint=2000.0, channels_hint=max(chs) + 1)

    # Start the subscriber before launching the UI
    sub.start()

    app = QtWidgets.QApplication([])
    w = WaveViewer(sub, channels=chs, win_s=args.win, downsample=args.downsample, ylabel=args.ylabel)
    w.resize(1100, 600)
    w.show()

    # Ensure subscriber stops when the app closes
    def _cleanup():
        try:
            sub.stop()
        except Exception:
            pass

    app.aboutToQuit.connect(_cleanup)
    app.exec_()


if __name__ == "__main__":
    main()
