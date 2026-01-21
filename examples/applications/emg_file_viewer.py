#!/usr/bin/env python3
"""
EMG File Viewer

Features implemented in this file:
- Waveform viewer with downsampling (min/max envelope)
- RMS heatmap tab
- File open for numpy (.npz/.npy/.csv) and simple .dat
- Play/pause with QTimer
- UI control for downsampling `max_points`

The code uses headless fallbacks when PyQt5/PySide2 or matplotlib are not present so
the module can be imported in CI environments.
"""
import sys
import os
import math
import numpy as np

# GUI availability
GUI_AVAILABLE = False
QtWidgets = None
QtCore = None
try:
    from PyQt5 import QtWidgets, QtCore
    GUI_AVAILABLE = True
except Exception:
    try:
        from PySide2 import QtWidgets, QtCore
        GUI_AVAILABLE = True
    except Exception:
        QtWidgets = None
        QtCore = None

# Matplotlib availability
MATPLOTLIB_AVAILABLE = False
Figure = None
FigureCanvas = None
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    class Figure:
        def clear(self):
            pass
        def add_subplot(self, *a, **k):
            class Ax:
                def clear(self): pass
                def plot(self, *a, **k): pass
                def text(self, *a, **k): pass
                def set_xlabel(self, *a, **k): pass
                def set_title(self, *a, **k): pass
                def imshow(self, *a, **k):
                    class Im: pass
                    return Im()
            return Ax()
        def tight_layout(self, *a, **k): pass
        def subplots_adjust(self, *a, **k): pass
        def add_axes(self, *a, **k):
            class C: pass
            return C()
        def colorbar(self, *a, **k): pass
    # if matplotlib present, import patches for overlays
    try:
        from matplotlib import patches as mpatches
    except Exception:
        mpatches = None

def downsample_for_plot(x, y, max_points=1):
    """Return an interleaved min/max envelope downsample of y with corresponding x.

    Preserves spikes by returning array of length ~2*n_out where each pair is (min, max).
    """
    # Treat max_points < 2 as "no downsampling" (plot raw data)
    if max_points is None or int(max_points) < 2:
        return x, y
    n = len(y)
    if n <= max_points:
        return x, y
    binsize = int(math.ceil(n / float(max_points)))
    n_out = int(math.ceil(n / float(binsize)))
    mins = np.empty(n_out)
    maxs = np.empty(n_out)
    xs = np.empty(n_out)
    for i in range(n_out):
        s = i * binsize
        e = min(n, (i + 1) * binsize)
        seg = y[s:e]
        mins[i] = np.min(seg)
        maxs[i] = np.max(seg)
        if x is not None:
            xs[i] = float(x[s + (e - s) // 2])
        else:
            xs[i] = float((s + e) / 2.0)
    # To avoid connecting the max of one bin to the min of the next (which creates
    # a sawtooth), insert NaNs between each (min,max) pair so matplotlib will
    # break the line between bins. Resulting length ~ n_out*3 - 1
    if n_out == 1:
        x_ds = np.array([xs[0], xs[0]])
        y_ds = np.array([mins[0], maxs[0]])
        return x_ds, y_ds
    x_ds = np.empty(n_out * 3 - 1)
    y_ds = np.empty(n_out * 3 - 1)
    for i in range(n_out):
        idx = i * 3
        x_ds[idx] = xs[i]
        y_ds[idx] = mins[i]
        x_ds[idx + 1] = xs[i]
        y_ds[idx + 1] = maxs[i]
        if i < n_out - 1:
            x_ds[idx + 2] = np.nan
            y_ds[idx + 2] = np.nan
    return x_ds, y_ds


def downsample_envelope(x, y, max_points=1):
    """Return bin centers `xs`, per-bin `mins` and `maxs` for envelope plotting.

    When `max_points < 2` the function returns the original `x` and `y` as
    three arrays where mins==maxs==y.
    """
    if max_points is None or int(max_points) < 2:
        return x, y, y
    n = len(y)
    if n <= max_points:
        return x, y, y
    binsize = int(math.ceil(n / float(max_points)))
    n_out = int(math.ceil(n / float(binsize)))
    mins = np.empty(n_out)
    maxs = np.empty(n_out)
    xs = np.empty(n_out)
    for i in range(n_out):
        s = i * binsize
        e = min(n, (i + 1) * binsize)
        seg = y[s:e]
        mins[i] = np.min(seg)
        maxs[i] = np.max(seg)
        if x is not None:
            xs[i] = float(x[s + (e - s) // 2])
        else:
            xs[i] = float((s + e) / 2.0)
    return xs, mins, maxs

# --- Visualization filter helpers (use intan realtime filter if available) ---
def _make_vis_filter(fs, n_channels, bp_low, bp_high, notch_freq, notch_q, enable_bp, enable_notch):
    try:
        from intan.processing import RealtimeFilter
    except Exception:
        return None
    try:
        rf = RealtimeFilter(
            fs=float(fs),
            n_channels=int(n_channels),
            bp_low=float(bp_low),
            bp_high=float(bp_high),
            enable_bandpass=bool(enable_bp),
            notch_freqs=(float(notch_freq),) if notch_freq else (),
            notch_q=float(notch_q),
            enable_notch=bool(enable_notch),
        )
        return rf
    except Exception:
        return None


def load_file_simple(path, num_channels=None):
    """Simple loader supporting .npz/.npy/.csv and whitespace-delimited .dat

    Returns samples x channels numpy array and sample rate if encoded (none otherwise).
    """
    ext = os.path.splitext(path)[1].lower()
    # try to support .rhd via intan.io if available
    try:
        from intan.io import load_rhd_file as _load_rhd
    except Exception:
        _load_rhd = None

    if ext in ('.npz', '.npz'):
        arr = np.load(path)
        # if archive contains 'data' key
        if isinstance(arr, np.lib.npyio.NpzFile) and 'data' in arr:
            data = arr['data']
        else:
            # try first array
            try:
                data = arr[arr.files[0]]
            except Exception:
                data = None
        if data is None:
            raise RuntimeError('No array found in npz')
    elif ext in ('.npy',):
        data = np.load(path)
    else:
        # csv, dat, or .rhd
        if ext == '.rhd' and _load_rhd is not None:
            res = _load_rhd(path)
            # try to extract amplifier data and time/sample rate
            if isinstance(res, dict):
                data = res.get('amplifier_data')
                if data is None:
                    # try first array-like entry
                    for k in list(res.keys()):
                        if hasattr(res[k], 'shape'):
                            data = res[k]; break
            else:
                data = None
            if data is None:
                raise RuntimeError('No amplifier data found in RHD file')
        else:
            try:
                data = np.loadtxt(path, delimiter=',')
            except Exception:
                data = np.loadtxt(path)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    # Heuristic to detect channel-major arrays (channels x samples) and convert to samples x channels
    s0, s1 = data.shape
    if num_channels is None:
        # If there are many more columns than rows and the row count is small (likely channel count),
        # assume data is channels x samples and transpose.
        if s0 < s1 and s0 <= 256 and s1 >= 2 * s0:
            data = data.T
    else:
        # If num_channels provided, handle common cases:
        # - If shape matches (channels, samples) -> transpose
        # - Else if samples are flattened with interleaved channels, try reshape
        if s0 == num_channels and s1 > 1:
            data = data.T
        elif s1 != num_channels and (data.shape[0] % num_channels == 0):
            # reshape from flattened samples to (n_samples, num_channels)
            data = data.reshape(-1, num_channels)
    return data

if GUI_AVAILABLE and MATPLOTLIB_AVAILABLE:
    class EMGViewer(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('EMG File Viewer')
            self.data = None
            self.fs = 2000.0
            self.current_pos = 0
            self.playing = False
            self.max_points = 1
            self.downsample_factor = 1

            # central widget and layout
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            vlay = QtWidgets.QVBoxLayout(central)

            # controls row (only file/open button)
            ctrl = QtWidgets.QHBoxLayout()
            btn_open = QtWidgets.QPushButton('Open')
            btn_open.clicked.connect(self.open_file)
            ctrl.addWidget(btn_open)
            ctrl.addStretch(1)
            vlay.addLayout(ctrl)

            # plot area: two tabs (Waveform, RMS Heatmap)
            self.max_points = 1
            self.plot_tabs = QtWidgets.QTabWidget()

            # Waveform tab
            wave_w = QtWidgets.QWidget()
            wave_layout = QtWidgets.QVBoxLayout(wave_w)
            self.fig = Figure(figsize=(10, 4))
            self.canvas = FigureCanvas(self.fig)
            wave_layout.addWidget(self.canvas, 1)
            self.plot_tabs.addTab(wave_w, 'Waveform')

            # RMS heatmap tab (layout note; full controls are in right-side params)
            heat_w = QtWidgets.QWidget()
            hlay = QtWidgets.QVBoxLayout(heat_w)
            # top hint: instruct user to use right-side params for layout/cmap
            hint = QtWidgets.QLabel('Configure RMS layout and colormap in the right-side Params panel')
            hint.setWordWrap(True)
            hlay.addWidget(hint)
            self.fig2 = Figure(figsize=(10, 4))
            self.canvas2 = FigureCanvas(self.fig2)
            hlay.addWidget(self.canvas2, 1)
            self.plot_tabs.addTab(heat_w, 'RMS Heatmap')

            # Channel QC tab
            qc_w = QtWidgets.QWidget()
            qc_l = QtWidgets.QVBoxLayout(qc_w)
            self.fig3 = Figure(figsize=(8, 3))
            self.canvas3 = FigureCanvas(self.fig3)
            qc_l.addWidget(self.canvas3, 1)
            self.qc_excluded_label = QtWidgets.QLabel('Excluded channels: None')
            qc_l.addWidget(self.qc_excluded_label)
            self.plot_tabs.addTab(qc_w, 'Channel QC')

            # build main content area: left=plots, right=scrollable params
            content_h = QtWidgets.QHBoxLayout()

            # left column: plot tabs + scrub + playback
            left_w = QtWidgets.QWidget()
            left_col = QtWidgets.QVBoxLayout(left_w)
            left_col.addWidget(self.plot_tabs, 1)

            # scrub slider
            self.scrub_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.scrub_slider.setMinimum(0)
            self.scrub_slider.setMaximum(0)
            self.scrub_slider.valueChanged.connect(self._on_slider_moved)
            try:
                self.scrub_slider.sliderPressed.connect(self._on_scrub_pressed)
                self.scrub_slider.sliderReleased.connect(self._on_scrub_released)
            except Exception:
                pass
            left_col.addWidget(self.scrub_slider)

            # playback controls row (below scrub)
            play_row = QtWidgets.QHBoxLayout()
            self.btn_restart = QtWidgets.QPushButton('<< Start')
            self.btn_restart.clicked.connect(self._go_start)
            play_row.addWidget(self.btn_restart)
            self.btn_play = QtWidgets.QPushButton('Play')
            self.btn_play.clicked.connect(self._toggle_play)
            play_row.addWidget(self.btn_play)
            play_row.addStretch(1)
            left_col.addLayout(play_row)

            content_h.addWidget(left_w, 1)

            # right column: scrollable params/config (grouped)
            params_widget = QtWidgets.QWidget()
            params_layout = QtWidgets.QVBoxLayout(params_widget)

            # --- General group ---
            gen_box = QtWidgets.QGroupBox('General')
            gen_l = QtWidgets.QVBoxLayout()
            gen_box.setLayout(gen_l)
            gen_l.addWidget(QtWidgets.QLabel('Channel'))
            # Use a dropdown so users can quickly pick a start channel by name/index
            self.chan_selector = QtWidgets.QComboBox()
            # keep a compact label showing how many channels are loaded
            self.lbl_channel_count = QtWidgets.QLabel('Channels loaded: 0')
            # connect change to replot; ignore the index argument
            try:
                self.chan_selector.currentIndexChanged.connect(lambda _idx: self.plot_waveforms())
            except Exception:
                pass
            gen_l.addWidget(self.chan_selector)
            gen_l.addWidget(self.lbl_channel_count)
            gen_l.addWidget(QtWidgets.QLabel('FS (Hz)'))
            self.spin_fs = QtWidgets.QSpinBox()
            self.spin_fs.setRange(1, 100000)
            self.spin_fs.setValue(int(self.fs))
            self.spin_fs.valueChanged.connect(self._on_fs_changed)
            gen_l.addWidget(self.spin_fs)
            gen_l.addWidget(QtWidgets.QLabel('Window (s)'))
            self.window_sec = QtWidgets.QDoubleSpinBox()
            self.window_sec.setRange(0.01, 60.0)
            self.window_sec.setSingleStep(0.01)
            self.window_sec.setValue(2.0)
            self.window_sec.valueChanged.connect(self.plot_waveforms)
            gen_l.addWidget(self.window_sec)
            gen_l.addWidget(QtWidgets.QLabel('Downsample (max points, 1 = no downsampling)'))
            self.spin_max_points = QtWidgets.QSpinBox()
            self.spin_max_points.setRange(1, 200000)
            self.spin_max_points.setValue(self.max_points)
            self.spin_max_points.valueChanged.connect(self._on_max_points_changed)
            gen_l.addWidget(self.spin_max_points)
            gen_l.addWidget(QtWidgets.QLabel('Decimate (integer factor, 1 = no decimation)'))
            self.spin_decimate = QtWidgets.QSpinBox()
            self.spin_decimate.setRange(1, 128)
            self.spin_decimate.setValue(1)
            self.spin_decimate.valueChanged.connect(self._on_decimate_changed)
            gen_l.addWidget(self.spin_decimate)
            self.autoscale_cb = QtWidgets.QCheckBox('Autoscale Y')
            self.autoscale_cb.setChecked(True)
            self.autoscale_cb.stateChanged.connect(self._on_autoscale_changed)
            gen_l.addWidget(self.autoscale_cb)
            gen_l.addWidget(QtWidgets.QLabel('Channels displayed'))
            self.spin_stack_count = QtWidgets.QSpinBox()
            self.spin_stack_count.setRange(1, 128)
            self.spin_stack_count.setValue(1)
            self.spin_stack_count.valueChanged.connect(self.plot_waveforms)
            gen_l.addWidget(self.spin_stack_count)
            params_layout.addWidget(gen_box)

            # --- RMS Heatmap group ---
            rms_box = QtWidgets.QGroupBox('RMS Heatmap')
            rms_l = QtWidgets.QVBoxLayout()
            rms_box.setLayout(rms_l)
            rms_l.addWidget(QtWidgets.QLabel('RMS Heatmap Layout'))
            self.combo_layout = QtWidgets.QComboBox()
            self.combo_layout.addItems(['8x8', '4x16', '16x4', 'Custom', '3D Cone'])
            rms_l.addWidget(self.combo_layout)
            rowcol_w = QtWidgets.QWidget()
            rowcol_l = QtWidgets.QHBoxLayout(rowcol_w)
            self.spin_rows = QtWidgets.QSpinBox(); self.spin_rows.setRange(1,128); self.spin_rows.setValue(8)
            self.spin_cols = QtWidgets.QSpinBox(); self.spin_cols.setRange(1,128); self.spin_cols.setValue(8)
            rowcol_l.addWidget(QtWidgets.QLabel('Rows')); rowcol_l.addWidget(self.spin_rows)
            rowcol_l.addWidget(QtWidgets.QLabel('Cols')); rowcol_l.addWidget(self.spin_cols)
            rms_l.addWidget(rowcol_w)
            rms_l.addWidget(QtWidgets.QLabel('Cmap'))
            self.cmap_combo = QtWidgets.QComboBox(); self.cmap_combo.addItems(['inferno','viridis','plasma','magma','cividis'])
            rms_l.addWidget(self.cmap_combo)
            self.btn_compute = QtWidgets.QPushButton('Compute RMS')
            self.btn_compute.clicked.connect(self.plot_heatmap)
            rms_l.addWidget(self.btn_compute)
            rms_l.addWidget(QtWidgets.QLabel('RMS window (ms)'))
            self.spin_rms_ms = QtWidgets.QSpinBox()
            self.spin_rms_ms.setRange(1, 5000)
            self.spin_rms_ms.setValue(100)
            rms_l.addWidget(self.spin_rms_ms)
            params_layout.addWidget(rms_box)

            # --- Filtering group ---
            filt_box = QtWidgets.QGroupBox('Filtering')
            filt_l = QtWidgets.QVBoxLayout()
            filt_box.setLayout(filt_l)
            self.chk_notch_enable = QtWidgets.QCheckBox('Enable notch filter')
            self.chk_notch_enable.setChecked(True)
            self.chk_notch_enable.stateChanged.connect(self.plot_waveforms)
            filt_l.addWidget(self.chk_notch_enable)
            notch_row = QtWidgets.QWidget()
            notch_l = QtWidgets.QHBoxLayout(notch_row)
            notch_l.addWidget(QtWidgets.QLabel('Notch (Hz)'))
            self.spin_notch_hz_vis = QtWidgets.QDoubleSpinBox(); self.spin_notch_hz_vis.setRange(10.0, 1000.0); self.spin_notch_hz_vis.setValue(60.0); self.spin_notch_hz_vis.setSingleStep(1.0)
            self.spin_notch_hz_vis.valueChanged.connect(self.plot_waveforms)
            notch_l.addWidget(self.spin_notch_hz_vis)
            notch_l.addWidget(QtWidgets.QLabel('Q'))
            self.spin_notch_q_vis = QtWidgets.QDoubleSpinBox(); self.spin_notch_q_vis.setRange(0.1, 100.0); self.spin_notch_q_vis.setValue(30.0); self.spin_notch_q_vis.setSingleStep(0.1)
            self.spin_notch_q_vis.valueChanged.connect(self.plot_waveforms)
            notch_l.addWidget(self.spin_notch_q_vis)
            filt_l.addWidget(notch_row)

            self.chk_bp_enable = QtWidgets.QCheckBox('Enable bandpass filter')
            self.chk_bp_enable.setChecked(True)
            self.chk_bp_enable.stateChanged.connect(self.plot_waveforms)
            filt_l.addWidget(self.chk_bp_enable)
            bp_row = QtWidgets.QWidget()
            bp_l = QtWidgets.QHBoxLayout(bp_row)
            bp_l.addWidget(QtWidgets.QLabel('BP low (Hz)'))
            self.spin_bp_low_vis = QtWidgets.QDoubleSpinBox(); self.spin_bp_low_vis.setRange(0.1, 1000.0); self.spin_bp_low_vis.setValue(10.0); self.spin_bp_low_vis.setSingleStep(1.0)
            self.spin_bp_low_vis.valueChanged.connect(self.plot_waveforms)
            bp_l.addWidget(self.spin_bp_low_vis)
            bp_l.addWidget(QtWidgets.QLabel('BP high (Hz)'))
            self.spin_bp_high_vis = QtWidgets.QDoubleSpinBox(); self.spin_bp_high_vis.setRange(1.0, 2000.0); self.spin_bp_high_vis.setValue(500.0); self.spin_bp_high_vis.setSingleStep(1.0)
            self.spin_bp_high_vis.valueChanged.connect(self.plot_waveforms)
            bp_l.addWidget(self.spin_bp_high_vis)
            filt_l.addWidget(bp_row)

            self.chk_apply_filters_to_rms = QtWidgets.QCheckBox('Apply filters to RMS')
            self.chk_apply_filters_to_rms.setChecked(False)
            self.chk_apply_filters_to_rms.stateChanged.connect(self.plot_heatmap)
            filt_l.addWidget(self.chk_apply_filters_to_rms)

            params_layout.addWidget(filt_box)

            # --- Channel QC group ---
            qc_box = QtWidgets.QGroupBox('Channel QC')
            qc_l = QtWidgets.QVBoxLayout()
            qc_box.setLayout(qc_l)
            qc_l.addWidget(QtWidgets.QLabel('QC window (ms)'))
            self.spin_qc_ms = QtWidgets.QSpinBox()
            self.spin_qc_ms.setRange(10, 5000)
            self.spin_qc_ms.setValue(200)
            qc_l.addWidget(self.spin_qc_ms)
            qc_l.addWidget(QtWidgets.QLabel('QC robust z warn'))
            self.spin_qc_z_warn = QtWidgets.QDoubleSpinBox()
            self.spin_qc_z_warn.setRange(0.1, 10.0); self.spin_qc_z_warn.setSingleStep(0.1)
            self.spin_qc_z_warn.setValue(2.0)
            qc_l.addWidget(self.spin_qc_z_warn)
            qc_l.addWidget(QtWidgets.QLabel('QC robust z bad'))
            self.spin_qc_z_bad = QtWidgets.QDoubleSpinBox()
            self.spin_qc_z_bad.setRange(0.1, 10.0); self.spin_qc_z_bad.setSingleStep(0.1)
            self.spin_qc_z_bad.setValue(3.0)
            qc_l.addWidget(self.spin_qc_z_bad)
            qc_l.addWidget(QtWidgets.QLabel('QC powerline ratio'))
            self.spin_qc_pl = QtWidgets.QDoubleSpinBox()
            self.spin_qc_pl.setRange(0.0, 1.0); self.spin_qc_pl.setSingleStep(0.01)
            self.spin_qc_pl.setValue(0.30)
            qc_l.addWidget(self.spin_qc_pl)
            qc_l.addWidget(QtWidgets.QLabel('QC flat std min'))
            self.spin_qc_flat = QtWidgets.QDoubleSpinBox()
            self.spin_qc_flat.setRange(0.0, 1000.0); self.spin_qc_flat.setSingleStep(0.1)
            self.spin_qc_flat.setValue(1.0)
            qc_l.addWidget(self.spin_qc_flat)
            qc_l.addWidget(QtWidgets.QLabel('QC zc min (Hz)'))
            self.spin_qc_zc = QtWidgets.QDoubleSpinBox()
            self.spin_qc_zc.setRange(0.0, 100.0); self.spin_qc_zc.setSingleStep(0.1)
            self.spin_qc_zc.setValue(3.0)
            qc_l.addWidget(self.spin_qc_zc)
            qc_l.addWidget(QtWidgets.QLabel('QC consec bad needed'))
            self.spin_qc_bad_cons = QtWidgets.QSpinBox(); self.spin_qc_bad_cons.setRange(1, 20); self.spin_qc_bad_cons.setValue(3)
            qc_l.addWidget(self.spin_qc_bad_cons)
            qc_l.addWidget(QtWidgets.QLabel('QC consec good needed'))
            self.spin_qc_good_cons = QtWidgets.QSpinBox(); self.spin_qc_good_cons.setRange(1, 50); self.spin_qc_good_cons.setValue(5)
            qc_l.addWidget(self.spin_qc_good_cons)
            qc_l.addWidget(QtWidgets.QLabel('QC notch (Hz)'))
            self.spin_qc_notch = QtWidgets.QDoubleSpinBox(); self.spin_qc_notch.setRange(30.0, 100.0); self.spin_qc_notch.setValue(60.0)
            qc_l.addWidget(self.spin_qc_notch)
            qc_l.addWidget(QtWidgets.QLabel('QC notch Q'))
            self.spin_qc_notchQ = QtWidgets.QDoubleSpinBox(); self.spin_qc_notchQ.setRange(1.0, 100.0); self.spin_qc_notchQ.setValue(30.0)
            qc_l.addWidget(self.spin_qc_notchQ)
            qc_l.addWidget(QtWidgets.QLabel('QC bp low (Hz)'))
            self.spin_qc_bp_low = QtWidgets.QDoubleSpinBox(); self.spin_qc_bp_low.setRange(0.1, 100.0); self.spin_qc_bp_low.setValue(10.0)
            qc_l.addWidget(self.spin_qc_bp_low)
            qc_l.addWidget(QtWidgets.QLabel('QC bp high (Hz)'))
            self.spin_qc_bp_high = QtWidgets.QDoubleSpinBox(); self.spin_qc_bp_high.setRange(10.0, 1000.0); self.spin_qc_bp_high.setValue(500.0)
            qc_l.addWidget(self.spin_qc_bp_high)
            # QC actions
            self.btn_qc_compute = QtWidgets.QPushButton('Compute QC')
            self.btn_qc_compute.clicked.connect(self.compute_qc)
            qc_l.addWidget(self.btn_qc_compute)
            qc_l.addWidget(QtWidgets.QLabel('QC metric'))
            self.combo_qc_metric = QtWidgets.QComboBox()
            self.combo_qc_metric.addItems(['robust_z', 'rms', 'pl_ratio'])
            qc_l.addWidget(self.combo_qc_metric)
            cal_w = QtWidgets.QWidget()
            cal_l = QtWidgets.QHBoxLayout(cal_w)
            self.btn_qc_calibrate = QtWidgets.QPushButton('Begin Cal')
            self.btn_qc_calibrate.clicked.connect(self._qc_begin_calibration)
            cal_l.addWidget(self.btn_qc_calibrate)
            self.btn_qc_finalize = QtWidgets.QPushButton('Finalize Cal')
            self.btn_qc_finalize.clicked.connect(self._qc_finalize_calibration)
            cal_l.addWidget(self.btn_qc_finalize)
            qc_l.addWidget(cal_w)
            self.btn_qc_export = QtWidgets.QPushButton('Export QC JSON')
            self.btn_qc_export.clicked.connect(self._qc_export_json)
            qc_l.addWidget(self.btn_qc_export)
            params_layout.addWidget(qc_box)

            params_layout.addStretch(1)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(params_widget)
            scroll.setFixedWidth(320)
            content_h.addWidget(scroll)

            vlay.addLayout(content_h)
            # timer for playback
            self.play_timer = QtCore.QTimer()
            self.play_timer.setInterval(50)
            self.play_timer.timeout.connect(self._on_timer_tick)

            # autoscale state
            self.autoscale = True

            self.resize(1000, 700)

        def _on_fs_changed(self, val):
            self.fs = float(val)
            try:
                self.plot_waveforms()
            except Exception:
                pass

        def _on_max_points_changed(self, val):
            self.max_points = int(val)
            self.plot_waveforms()

        def _on_decimate_changed(self, val):
            try:
                self.downsample_factor = max(1, int(val))
            except Exception:
                self.downsample_factor = 1
            try:
                self.plot_waveforms()
            except Exception:
                pass

        def _on_autoscale_changed(self, state):
            try:
                self.autoscale = bool(int(state))
            except Exception:
                self.autoscale = bool(state)
            try:
                self.plot_waveforms()
            except Exception:
                pass

        def open_file(self):
            # Determine initial directory: env var -> user config file (~/.emg_viewer_config) -> fallback
            initial_dir = os.environ.get('EMG_VIEWER_DEFAULT_DIR')
            if not initial_dir:
                cfg_path = os.path.expanduser('~/.emg_viewer_config')
                if os.path.exists(cfg_path):
                    try:
                        import json
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                        if isinstance(cfg, dict):
                            initial_dir = cfg.get('default_dir')
                    except Exception:
                        initial_dir = None
            if not initial_dir:
                initial_dir = r"G:\\Shared drives\\NML_shared\\DataShare\\HDEMG Human Healthy\\HD-EMG_Cuff\\Jonathan\\2025_09_15\\raw\\exo_gestures_v3_250915_195637"

            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open EMG file', initial_dir,
                                                           'EMG Files (*.npz *.npy *.csv *.dat *.rhd);;RHD Files (*.rhd);;All Files (*)')
            if not path:
                return
            # Try to load with intan loader to preserve metadata if possible
            loaded_meta = None
            data_arr = None
            try:
                from intan.io import load_rhd_file
                if path.lower().endswith('.rhd'):
                    res = load_rhd_file(path, verbose=False)
                    if isinstance(res, dict) and 'amplifier_data' in res:
                        # amplifier_data is channels x samples -> transpose to samples x channels
                        a = res['amplifier_data']
                        data_arr = np.asarray(a).T
                        loaded_meta = res
            except Exception:
                loaded_meta = None
            if data_arr is None:
                try:
                    arr = load_file_simple(path)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Error', str(e))
                    return
                data_arr = np.asarray(arr)
            self.data = data_arr
            if self.data.ndim == 1:
                self.data = self.data[:, None]
            # attempt to extract units and channel names from loader metadata
            self.units = 'µV'  # sensible default for EMG from Intan
            try:
                if loaded_meta is not None:
                    # Intan loader scales amplifier_data to microvolts
                    self.units = 'µV'
                    chn = loaded_meta.get('channel_names') or loaded_meta.get('amplifier_channels')
                    if isinstance(chn, (list, tuple)) and len(chn) == self.data.shape[1]:
                        try:
                            # if amplifier_channels entries are dicts with 'native_channel_name'
                            names = [c.get('native_channel_name') if isinstance(c, dict) else str(c) for c in chn]
                            self.channel_names = names
                        except Exception:
                            self.channel_names = None
                else:
                    # try to detect npz metadata
                    try:
                        if path.lower().endswith('.npz'):
                            d = np.load(path)
                            if 'units' in d:
                                self.units = str(d['units'])
                            elif 'amplifier_units' in d:
                                self.units = str(d['amplifier_units'])
                    except Exception:
                        pass
            except Exception:
                self.units = 'µV'
            # populate channel selector with channel names (if available) or numeric labels
            try:
                self.chan_selector.blockSignals(True)
                self.chan_selector.clear()
                n_ch = max(1, self.data.shape[1])
                names = getattr(self, 'channel_names', None)
                for i in range(n_ch):
                    if names and i < len(names) and names[i]:
                        label = f"{i+1}: {names[i]}"
                    else:
                        label = f"Ch {i+1}"
                    # store zero-based channel index as item data
                    try:
                        self.chan_selector.addItem(label, i)
                    except Exception:
                        # fallback if addItem signature differs
                        self.chan_selector.addItem(label)
                # update channel count label
                try:
                    self.lbl_channel_count.setText(f"Channels loaded: {n_ch}")
                except Exception:
                    pass
            finally:
                try:
                    self.chan_selector.blockSignals(False)
                except Exception:
                    pass
            self.scrub_slider.setMaximum(max(0, self.data.shape[0] - 1))
            self.current_pos = 0
            self.plot_waveforms()

        def _go_start(self):
            """Jump playback back to start and update UI."""
            self.current_pos = 0
            try:
                self.scrub_slider.blockSignals(True)
                self.scrub_slider.setValue(0)
            finally:
                try:
                    self.scrub_slider.blockSignals(False)
                except Exception:
                    pass
            try:
                self.plot_waveforms()
            except Exception:
                pass

        def plot_waveforms(self):
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            if self.data is None:
                ax.text(0.5, 0.5, 'Open a file to begin', ha='center', va='center')
                self.canvas.draw_idle()
                return
            # determine start channel from dropdown (stored zero-based in item data)
            try:
                data = self.chan_selector.currentData()
                if data is None:
                    start_chan = max(0, min(0, self.data.shape[1]-1))
                else:
                    start_chan = int(data)
            except Exception:
                try:
                    # fallback to textual parsing
                    start_chan = int(str(self.chan_selector.currentText()).split()[0].rstrip(':')) - 1
                except Exception:
                    start_chan = 0
            stack_count = int(getattr(self, 'spin_stack_count', None).value()) if getattr(self, 'spin_stack_count', None) is not None else 1
            fs = max(1.0, float(self.spin_fs.value()))
            window_s = float(self.window_sec.value())
            window_samples = max(1, int(window_s * fs))
            start = int(self.current_pos)
            end = min(self.data.shape[0], start + window_samples)
            t = np.arange(start, end) / fs
            # clamp channel range
            start_chan = max(0, min(start_chan, max(0, self.data.shape[1]-1)))
            end_chan = min(self.data.shape[1], start_chan + max(1, stack_count))
            block = self.data[start:end, start_chan:end_chan]
            if block.size == 0 or block.shape[0] == 0:
                ax.text(0.5, 0.5, 'No data in window', ha='center', va='center')
                self.canvas.draw_idle()
                return
            C = block.shape[1]
            # (single-channel plotting handled below; multi-channel processed later)
            # apply visualization filters if enabled (for multi-channel too)
            try:
                if (getattr(self, 'chk_notch_enable', None) and self.chk_notch_enable.isChecked()) or (getattr(self, 'chk_bp_enable', None) and self.chk_bp_enable.isChecked()):
                    try:
                        bp_low = float(getattr(self, 'spin_bp_low_vis', None).value()) if getattr(self, 'spin_bp_low_vis', None) is not None else 10.0
                        bp_high = float(getattr(self, 'spin_bp_high_vis', None).value()) if getattr(self, 'spin_bp_high_vis', None) is not None else (fs/2.0 - 1.0)
                        notch_hz = float(getattr(self, 'spin_notch_hz_vis', None).value()) if getattr(self, 'spin_notch_hz_vis', None) is not None else None
                        notch_q = float(getattr(self, 'spin_notch_q_vis', None).value()) if getattr(self, 'spin_notch_q_vis', None) is not None else 30.0
                        visf = _make_vis_filter(fs, C, bp_low, bp_high, notch_hz, notch_q, getattr(self, 'chk_bp_enable').isChecked(), getattr(self, 'chk_notch_enable').isChecked())
                        if visf is not None:
                            try:
                                proc = visf.process(block.T)
                                block = proc.T
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

            # apply integer decimation (subsample) to reduce displayed samples
            try:
                dec = int(getattr(self, 'downsample_factor', 1))
            except Exception:
                dec = 1
            if dec > 1:
                try:
                    block = block[::dec, :]
                    t = t[::dec]
                except Exception:
                    pass

            # Plot stacked channels
            if C == 1:
                y = block[:, 0]
                tx_ty = downsample_for_plot(t, y, max_points=self.max_points)
                try:
                    tx, ty = tx_ty
                    ax.plot(tx, ty, lw=0.8)
                except Exception:
                    ax.plot(t, y, lw=0.8)
                # highlight if this channel is currently marked excluded by QC
                try:
                    if getattr(self, 'qc_instance', None) is not None:
                        excluded = getattr(self.qc_instance, '_is_bad', None)
                        if excluded is not None and len(excluded) > start_chan and excluded[start_chan]:
                            ax.set_facecolor('#fff0f0')
                            ax.text(0.02, 0.95, 'EXCLUDED', transform=ax.transAxes, color='red', fontsize=10, weight='bold')
                except Exception:
                    pass
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'Amplitude ({getattr(self, "units", "µV")})')
                ax.set_title(f'Channel {start_chan+1}  [{start}:{end}]')
                if not getattr(self, 'autoscale', True):
                    try:
                        full_amp = float(np.max(np.abs(self.data[:, start_chan])))
                        if full_amp > 0:
                            ax.set_ylim(-full_amp, full_amp)
                    except Exception:
                        pass
            else:
                # downsample each channel independently and stack
                ys_ds = []
                xs_ds = []
                max_amp = 0.0
                for ci in range(C):
                    try:
                        tx, ty = downsample_for_plot(t, block[:, ci], max_points=self.max_points)
                    except Exception:
                        tx, ty = t, block[:, ci]
                    xs_ds.append(tx)
                    ys_ds.append(ty)
                    max_amp = max(max_amp, float(np.max(np.abs(ty))))
                spacing = max_amp * 2.5 if max_amp > 0 else 1.0
                offsets = np.arange(C)[::-1] * spacing
                for i in range(C):
                    ax.plot(xs_ds[i], ys_ds[i] + offsets[i], lw=0.8)
                # set y ticks and labels to channel indices/names
                yticks = offsets.tolist()
                ylabels = []
                for i in range(C):
                    idx = start_chan + i
                    if getattr(self, 'channel_names', None) and len(self.channel_names) > idx:
                        ylabels.append(str(self.channel_names[idx]))
                    else:
                        ylabels.append(f'CH{idx+1}')
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'Amplitude ({getattr(self, "units", "µV")})')
                ax.set_title(f'Channels {start_chan+1}-{end_chan}  [{start}:{end}]')
                ax.set_ylim(-spacing, offsets[0] + spacing)
            # Highlight isolated channels after QC with a red overlay in waveform view
            if getattr(self, 'qc_instance', None) is not None:
                excluded = getattr(self.qc_instance, '_is_bad', None)
                if excluded is not None:
                    for i, is_bad in enumerate(excluded):
                        if is_bad and start_chan <= i < end_chan:
                            channel_idx = i - start_chan
                            ax.axvspan(t[0], t[-1], ymin=channel_idx / C, ymax=(channel_idx + 1) / C, color='red', alpha=0.2)
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def plot_heatmap(self):
            self.fig2.clear()
            ax = self.fig2.add_subplot(111)
            if self.data is None:
                ax.text(0.5, 0.5, 'Open a file to compute RMS', ha='center', va='center')
                self.canvas2.draw_idle()
                return
            layout = self.combo_layout.currentText()
            if layout == 'Custom':
                rows = int(self.spin_rows.value())
                cols = int(self.spin_cols.value())
            else:
                if layout == '8x8': rows, cols = 8, 8
                elif layout == '4x16': rows, cols = 4, 16
                elif layout == '16x4': rows, cols = 16, 4
                elif layout == '3D Cone': rows, cols = 8, 16
                else: rows, cols = 8, 8
            # compute RMS per channel over configured window (centered at current_pos)
            try:
                fs = float(max(1.0, float(self.spin_fs.value())))
                rms_ms = int(getattr(self, 'spin_rms_ms').value()) if getattr(self, 'spin_rms_ms', None) is not None else 100
                win_samps = max(1, int((rms_ms / 1000.0) * fs))
                center = int(self.current_pos)
                start = max(0, center - win_samps // 2)
                end = min(self.data.shape[0], start + win_samps)
                if self.chk_apply_filters_to_rms.isChecked() and (getattr(self, 'chk_bp_enable', None) and self.chk_bp_enable.isChecked() or (getattr(self, 'chk_notch_enable', None) and self.chk_notch_enable.isChecked())):
                    # apply visualization filters to the selected window or full data for RMS
                    try:
                        # prepare block (N, C) -> (C, N)
                        block = self.data[start:end, :]
                        if block.shape[0] < 1:
                            arr = self.data
                        else:
                            arr = block
                        # use vis filter but ensure correct channel count
                        fs = float(max(1.0, float(self.spin_fs.value())))
                        n_ch = int(self.data.shape[1])
                        notch_hz = float(getattr(self, 'spin_notch_hz_vis', None).value()) if getattr(self, 'spin_notch_hz_vis', None) is not None else None
                        notch_q = float(getattr(self, 'spin_notch_q_vis', None).value()) if getattr(self, 'spin_qc_notchQ', None) is not None else 30.0
                        bp_low = float(getattr(self, 'spin_bp_low_vis', None).value()) if getattr(self, 'spin_bp_low_vis', None) is not None else 10.0
                        bp_high = float(getattr(self, 'spin_bp_high_vis', None).value()) if getattr(self, 'spin_bp_high_vis', None) is not None else (fs/2.0 - 1.0)
                        visf = _make_vis_filter(fs, n_ch, bp_low, bp_high, notch_hz, notch_q, getattr(self, 'chk_bp_enable').isChecked(), getattr(self, 'chk_notch_enable').isChecked())
                        if visf is not None:
                            try:
                                proc = visf.process(arr.T)
                                arr2 = proc.T
                            except Exception:
                                arr2 = arr
                        else:
                            arr2 = arr
                        if arr2.shape[0] < 1:
                            rms_mean = np.sqrt(np.mean(self.data**2, axis=0))
                        else:
                            rms_mean = np.sqrt(np.mean(arr2**2, axis=0))
                    except Exception:
                        rms_mean = np.sqrt(np.mean(self.data**2, axis=0))
                else:
                    if end - start < 1:
                        rms_mean = np.sqrt(np.mean(self.data**2, axis=0))
                    else:
                        rms_mean = np.sqrt(np.mean(self.data[start:end, :]**2, axis=0))
            except Exception:
                # fallback to global RMS
                rms_mean = np.sqrt(np.mean(self.data**2, axis=0))
            grid = np.full((rows, cols), np.nan)
            nch = self.data.shape[1]
            for i in range(min(nch, rows * cols)):
                r = i // cols
                c = i % cols
                val = float(rms_mean[i])
                # gray out excluded channels
                try:
                    if getattr(self, 'qc_instance', None) is not None and self.qc_instance._is_bad[i]:
                        grid[r, c] = np.nan
                        # store excluded marker via negative inf to later annotate
                        # we'll use nan so color shows empty; also add overlay markers below
                    else:
                        grid[r, c] = val
                except Exception:
                    grid[r, c] = val
            cmap = str(self.cmap_combo.currentText())
            im = ax.imshow(grid, cmap=cmap, interpolation='nearest', aspect='auto')
            # overlay excluded channel markers
            try:
                if getattr(self, 'qc_instance', None) is not None:
                    bad_inds = [int(i) for i in np.where(self.qc_instance._is_bad)[0]]
                    for bi in bad_inds:
                        if bi < rows * cols:
                            rr = bi // cols; cc = bi % cols
                            try:
                                if 'mpatches' in globals() and mpatches is not None:
                                    rect = mpatches.Rectangle((cc-0.5, rr-0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
                                    ax.add_patch(rect)
                                else:
                                    import matplotlib.patches as _mp
                                    rect = _mp.Rectangle((cc-0.5, rr-0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
                                    ax.add_patch(rect)
                            except Exception:
                                pass
            except Exception:
                pass
            ax.set_title('RMS heatmap')
            self.fig2.subplots_adjust(right=0.85)
            cax = self.fig2.add_axes([0.88, 0.15, 0.03, 0.7])
            self.fig2.colorbar(im, cax=cax)
            self.fig2.tight_layout(rect=[0, 0, 0.85, 1])
            self.canvas2.draw_idle()

            # Enable interactivity for RMS heatmap
            def on_click(event):
                if event.xdata is not None and event.ydata is not None:
                    row = int(event.ydata)
                    col = int(event.xdata)
                    print(f"Clicked on row {row}, column {col}")

            self.fig2.canvas.mpl_connect('button_press_event', on_click)

        def compute_qc(self):
            """Compute channel quality over a window centered at current_pos and plot results."""
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'QC', 'No data loaded')
                return
            try:
                fs = float(max(1.0, float(self.spin_fs.value())))
                qc_ms = int(getattr(self, 'spin_qc_ms').value()) if getattr(self, 'spin_qc_ms', None) is not None else 200
                window_sec = max(0.001, qc_ms / 1000.0)
                n_channels = int(self.data.shape[1])
                from intan.processing._channel_qc import ChannelQC, QCParams
                params = QCParams()
                # override params from UI if present
                try:
                    params.robust_z_warn = float(self.spin_qc_z_warn.value())
                    params.robust_z_bad = float(self.spin_qc_z_bad.value())
                    params.pl_ratio_thresh = float(self.spin_qc_pl.value())
                    params.flat_std_min = float(self.spin_qc_flat.value())
                    params.zc_min_hz = float(self.spin_qc_zc.value())
                    params.consec_bad_needed = int(self.spin_qc_bad_cons.value())
                    params.consec_good_needed = int(self.spin_qc_good_cons.value())
                    params.notch_hz = float(self.spin_qc_notch.value())
                    params.notch_Q = float(self.spin_qc_notchQ.value())
                    params.bp_low_hz = float(self.spin_qc_bp_low.value())
                    params.bp_high_hz = float(self.spin_qc_bp_high.value())
                except Exception:
                    pass

                # cache ChannelQC instance so calibration persists
                if getattr(self, 'qc_instance', None) is None:
                    self.qc_instance = ChannelQC(fs=int(fs), n_channels=n_channels, window_sec=window_sec, params=params)
                else:
                    # if sampling window changed, recreate instance
                    if abs(self.qc_instance.window_sec - window_sec) > 1e-6 or self.qc_instance.n_channels != n_channels:
                        self.qc_instance = ChannelQC(fs=int(fs), n_channels=n_channels, window_sec=window_sec, params=params)
                    else:
                        # update params
                        self.qc_instance.params = params
                qc = self.qc_instance
                # Extract window centered at current_pos
                center = int(self.current_pos)
                win_samps = max(1, int(window_sec * fs))
                start = max(0, center - win_samps // 2)
                end = min(self.data.shape[0], start + win_samps)
                chunk = self.data[start:end, :]
                qc.update(chunk)
                out = qc.evaluate(compute_psd=True)

                metrics = out.get('metrics', {})
                bad = out.get('bad')
                watch = out.get('watch')
                excluded = out.get('excluded', set())

                # select metric to plot
                metric = str(self.combo_qc_metric.currentText()) if getattr(self, 'combo_qc_metric', None) is not None else 'robust_z'
                self.fig3.clear()
                ax = self.fig3.add_subplot(111)
                n = n_channels
                vals = metrics.get(metric, np.zeros(n))
                x = np.arange(n)
                colors = ['tab:blue'] * n
                for i in range(n):
                    if i in excluded:
                        colors[i] = 'red'
                    elif watch is not None and watch[i]:
                        colors[i] = 'orange'
                ax.bar(x, vals, color=colors)
                ax.set_xlabel('Channel')
                ax.set_ylabel(f'{metric}')
                ax.set_title(f'Channel QC @ {start}:{end} ({win_samps} samples)')
                self.fig3.tight_layout()
                self.canvas3.draw_idle()

                # update excluded label
                try:
                    txt = ', '.join(map(str, sorted(list(excluded)))) if excluded else 'None'
                    self.qc_excluded_label.setText(f'Excluded channels: {txt}')
                except Exception:
                    pass
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'QC Error', str(e))

        def _on_timer_tick(self):
            if self.data is None:
                return
            fs = float(max(1.0, float(self.spin_fs.value())))
            dt = self.play_timer.interval() / 1000.0
            step = max(1, int(fs * dt))
            self.current_pos = int(self.current_pos) + step
            window_s = float(self.window_sec.value())
            window_samples = max(1, int(window_s * fs))
            max_pos = max(0, self.data.shape[0] - window_samples)
            if self.current_pos >= max_pos:
                self.current_pos = max_pos
                self.play_timer.stop()
                self.playing = False
                self.btn_play.setText('Play')
            # update plot
            # update scrub slider without emitting signals
            try:
                self.scrub_slider.blockSignals(True)
                self.scrub_slider.setValue(int(self.current_pos))
            finally:
                try: self.scrub_slider.blockSignals(False)
                except Exception: pass
            if self.plot_tabs.currentIndex() == 0:
                self.plot_waveforms()
            elif self.plot_tabs.currentIndex() == 1:
                self.plot_heatmap()
            else:
                # QC tab
                try:
                    self.compute_qc()
                except Exception:
                    pass

        def _toggle_play(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Play', 'No data loaded')
                return
            if self.playing:
                self.play_timer.stop()
                self.playing = False
                self.btn_play.setText('Play')
            else:
                try:
                    self.scrub_slider.setMaximum(max(0, self.data.shape[0]-1))
                except Exception:
                    pass
                self.play_timer.start()
                self.playing = True
                self.btn_play.setText('Pause')

        def _qc_begin_calibration(self):
            try:
                if getattr(self, 'qc_instance', None) is not None:
                    self.qc_instance.begin_calibration()
                    QtWidgets.QMessageBox.information(self, 'QC', 'Calibration: begin - will lock on next evaluate()')
                else:
                    QtWidgets.QMessageBox.information(self, 'QC', 'No QC instance yet - compute QC once to initialize')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'QC Error', str(e))

        def _qc_finalize_calibration(self):
            try:
                if getattr(self, 'qc_instance', None) is not None:
                    self.qc_instance.finalize_calibration()
                    QtWidgets.QMessageBox.information(self, 'QC', 'Calibration finalized and locked')
                else:
                    QtWidgets.QMessageBox.information(self, 'QC', 'No QC instance yet - compute QC once to initialize')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'QC Error', str(e))

        def _qc_export_json(self):
            try:
                if getattr(self, 'qc_instance', None) is None:
                    QtWidgets.QMessageBox.information(self, 'QC', 'No QC data to export')
                    return
                out = self.qc_instance.evaluate(compute_psd=True)
                metrics = out.get('metrics', {})
                bad = [int(x) for x in np.where(out.get('bad', np.zeros(len(metrics.get('rms', [])), dtype=bool)))[0]]
                payload = {
                    'metrics': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in metrics.items()},
                    'bad': bad,
                    'watch': [int(x) for x in np.where(out.get('watch', np.zeros(len(metrics.get('rms', [])), dtype=bool)))[0]]
                }
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export QC JSON', '', 'JSON Files (*.json);;All Files (*)')
                if not path:
                    return
                import json
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
                QtWidgets.QMessageBox.information(self, 'QC', f'QC exported: {path}')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'QC Error', str(e))

        def _on_scrub_pressed(self):
            self._was_playing_on_scrub = self.playing
            if self.playing:
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')

        def _on_scrub_released(self):
            if getattr(self, '_was_playing_on_scrub', False):
                self.play_timer.start(); self.playing = True; self.btn_play.setText('Pause')

        def _on_slider_moved(self, val):
            self.current_pos = int(val)
            # Update waveform or heatmap depending on active tab
            try:
                if getattr(self, 'plot_tabs', None) is not None and self.plot_tabs.currentIndex() != 0:
                    self.plot_heatmap()
                else:
                    self.plot_waveforms()
            except Exception:
                try:
                    self.plot_waveforms()
                except Exception:
                    pass

    def main():
        app = QtWidgets.QApplication(sys.argv)
        viewer = EMGViewer()
        viewer.show()
        sys.exit(app.exec_())

else:
    class EMGViewer:
        def __init__(self):
            self.data = None
            self.fs = 2000.0
            self.current_pos = 0
            self.max_points = 2000
        def show(self):
            print('Headless: EMGViewer instantiated (Qt or matplotlib not available)')

    def main():
        v = EMGViewer()
        v.show()

if __name__ == '__main__':
    main()
