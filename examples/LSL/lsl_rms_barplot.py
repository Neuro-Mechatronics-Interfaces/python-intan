#!/usr/bin/env python3
"""
Real-time EMG RMS visualization with adaptive channel quality control.

Subscribes to an LSL EMG stream and displays RMS amplitude per channel with
automated quality checks (power line noise, flat channels, z-score outliers).

Usage:
    python lsl_rms_barplot.py --channels 128 --stream-type EMG --window-ms 200

Keyboard shortcuts:
    C: Calibrate QC (capture robust baseline from current state)
    E: Export list of excluded channels to excluded_channels.json
"""
import sys
import json
import argparse
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Use intan's LSL subscriber for consistent API
from intan.interface import LSLSubscriber, LSLStreamSpec
from intan.processing import ChannelQC, QCParams


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Real-time EMG RMS with adaptive channel QC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--stream-type", default="EMG", help="LSL stream type to subscribe to")
    ap.add_argument("--stream-name", default=None, help="LSL stream name (optional)")
    ap.add_argument("--channels", type=int, default=128, help="Number of channels to expect")
    ap.add_argument("--fs", type=float, default=1000.0, help="Expected sampling rate (Hz)")
    ap.add_argument("--window-ms", type=float, default=200, help="RMS window length (ms)")
    ap.add_argument("--update-ms", type=int, default=200, help="UI update interval (ms)")
    ap.add_argument("--ymax", type=float, default=500.0, help="Initial Y-axis max (auto-scales)")
    ap.add_argument("--timeout", type=float, default=5.0, help="LSL stream resolve timeout (s)")
    ap.add_argument("--verbose", action="store_true", help="Print connection info")
    return ap.parse_args()


def main():
    args = parse_args()

    # =========================
    # Config
    # =========================
    fs = args.fs
    n_channels = args.channels
    window_sec = args.window_ms / 1000.0
    ui_update_ms = args.update_ms
    ymax = args.ymax

    # Thresholds & behavior (tune here, not in the UI loop)
    qc_params = QCParams(
        robust_z_warn=2.0,
        robust_z_bad=3.0,
        pl_ratio_thresh=0.30,   # 60 Hz band / 20–450 Hz
        flat_std_min=1.0,       # adjust to your units
        zc_min_hz=3.0,
        consec_bad_needed=3,
        consec_good_needed=5,
        psd_min_hz=20.0,
        psd_max_hz=450.0,
        pl_low=58.0,
        pl_high=62.0,
        psd_every_n_evals=2,    # do PSD every N evaluate() calls
        bp_low_hz=10.0,
        bp_high_hz=450.0 if fs == 1000 else 500.0,  # safe top end
    )

    # =========================
    # LSL setup using intan API
    # =========================
    print(f"Looking for {args.stream_type} stream...")
    spec = LSLStreamSpec(
        name=args.stream_name,
        stype=args.stream_type,
        timeout=args.timeout,
        verbose=args.verbose
    )
    subscriber = LSLSubscriber(spec, fs_hint=fs, channels_hint=n_channels)
    subscriber.start()
    print(f"Connected to {args.stream_type} stream!")

    # =========================
    # QC engine
    # =========================
    qc = ChannelQC(fs=fs, n_channels=n_channels, window_sec=window_sec, params=qc_params)

    # =========================
    # UI
    # =========================
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Real-Time EMG Channel QC (Adaptive)")
    win.resize(1600, 600)

    # Top plot: RMS bars
    plot = win.addPlot(title="EMG RMS per Channel (adaptive QC)")
    xpos = np.arange(n_channels)
    heights = np.zeros(n_channels, dtype=float)

    bar_item = pg.BarGraphItem(x=xpos, height=heights, width=0.8, brush='dodgerblue')
    plot.addItem(bar_item)
    plot.setYRange(0, ymax)
    plot.setXRange(-1, n_channels)
    plot.setLabel('left', "RMS Amplitude")
    plot.setLabel('bottom', "Channel")

    # Bottom row: status text
    win.nextRow()
    status_lbl = pg.LabelItem(justify='left')
    win.addItem(status_lbl)

    win.show()

    # Keep a reference so we can replace it efficiently
    bar_ref = {'item': bar_item}

    # =========================
    # Helpers
    # =========================
    def color_brushes(is_bad_mask: np.ndarray, watch_mask: np.ndarray):
        """Return a list of per-bar brushes based on masks."""
        brushes = []
        for i in range(n_channels):
            if is_bad_mask[i]:
                brushes.append('red')
            elif watch_mask[i]:
                brushes.append('orange')
            else:
                brushes.append('dodgerblue')
        return brushes

    def export_excluded(excluded_set):
        arr = sorted(int(i) for i in excluded_set)
        print(f"[QC] Excluded channels: {arr}")
        # Also drop a small JSON file next to the script (optional)
        try:
            with open("excluded_channels.json", "w") as f:
                json.dump(arr, f, indent=2)
            print("[QC] Saved excluded channels to excluded_channels.json")
        except Exception as e:
            print(f"[QC] Could not save excluded list: {e}")

    # =========================
    # Update loop
    # =========================
    def update():
        # 1) Pull chunk from LSL using intan's subscriber API
        X = subscriber.read_window(window_sec)  # (T, C)
        if X.size > 0:
            qc.update(X)  # (samples, n_channels)

        # 2) Evaluate quality
        out = qc.evaluate()               # all heavy-lifting done inside
        excluded = out['excluded']
        metrics = out['metrics']
        rms_vals = metrics['rms']
        watch_mask = out['watch']         # bool array length n_channels

        # Build is_bad_mask to align with watch_mask
        is_bad_mask = np.zeros(n_channels, dtype=bool)
        if excluded:
            is_bad_mask[list(excluded)] = True

        # 3) Update bars efficiently
        #    (BarGraphItem doesn't support per-bar brushes updates in-place,
        #     so we replace the item each frame—128 bars is fine at 5 Hz)
        brushes = color_brushes(is_bad_mask, watch_mask)

        plot.removeItem(bar_ref['item'])
        new_bar = pg.BarGraphItem(x=xpos, height=np.nan_to_num(rms_vals, nan=0.0),
                                  width=0.8, brushes=brushes)
        plot.addItem(new_bar)
        bar_ref['item'] = new_bar

        # 4) Adaptive y-scale (cap at 5× base)
        valid = np.isfinite(rms_vals)
        if np.any(valid):
            target_ymax = max(ymax, float(np.percentile(rms_vals[valid], 99) * 1.2))
            plot.setYRange(0, min(target_ymax, ymax * 5.0))

        # 5) Status text
        status_lbl.setText(
            f"<b>Excluded:</b> {sorted(excluded)}  "
            f"&nbsp;&nbsp; <b>z_bad:</b> {qc_params.robust_z_bad:.1f} "
            f"&nbsp;&nbsp; <b>PL&gt;{qc_params.pl_ratio_thresh:.2f}</b> "
            f"&nbsp;&nbsp; <b>Flat std&lt;{qc_params.flat_std_min:.1f}</b> "
            f"&nbsp;&nbsp; <b>ZC&lt;{qc_params.zc_min_hz:.1f} Hz</b>"
        )

    # =========================
    # Hotkeys
    # =========================
    def on_key(event):
        key = event.key()
        if key in (QtCore.Qt.Key_C,):  # (Re)calibrate: lock robust center/scale from current state
            print("[QC] Calibration: capturing robust center/scale on next evaluate()...")
            qc.begin_calibration()
            # Let a few updates run, then finalize on next cycle
            QtCore.QTimer.singleShot(500, lambda: (qc.finalize_calibration(),
                                                   print("[QC] Calibration finalized.")))
        elif key in (QtCore.Qt.Key_E,):  # Export excluded channels
            out = qc.evaluate(compute_psd=False)
            export_excluded(out['excluded'])

    win.keyPressEvent = on_key

    # =========================
    # Timer
    # =========================
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(ui_update_ms)

    # Ensure clean shutdown
    def cleanup():
        try:
            subscriber.stop()
        except Exception:
            pass

    app.aboutToQuit.connect(cleanup)

    # =========================
    # Run
    # =========================
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
