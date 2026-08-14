"""
intan.processing._emg_viewer_gui

Graphical User Interface (GUI) for loading and visualizing EMG data.

The EMGViewer provides an interactive interface for exploring high-density EMG signals
from Intan `.rhd` recordings. It supports:
- Channel selection
- Segment visualization
- Preprocessing
- Trigger-based analysis (if available)

Typically launched via `_emg_launcher.py`.
"""

import os
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
try:
    from scipy.signal import spectrogram, butter, filtfilt, iirnotch
except Exception:
    spectrogram = butter = filtfilt = iirnotch = None
try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception:
    tk = None
    filedialog = None
    ttk = None

try:
    from intan.io import load_rhd_file
except Exception:
    load_rhd_file = None

try:
    from intan.processing import (
        bandpass_filter,
        lowpass_filter,
        notch_filter,
        rectify,
        window_rms,
        extract_features,
        common_average_reference,
        envelope_extraction,
    )
except Exception:
    bandpass_filter = lowpass_filter = notch_filter = rectify = None
    window_rms = extract_features = common_average_reference = envelope_extraction = None

# Visualization libraries (guarded)
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:
    plt = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None

# Try to provide a PyQt5-based viewer as modern alternative; fall back to Tkinter below
try:
    from PyQt5 import QtWidgets, QtCore
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvasQt
    from matplotlib.figure import Figure as MplFigure
    _HAS_PYQT = True
except Exception:
    _HAS_PYQT = False

def _downsample_for_plot(y, max_points=2000):
    import math
    n = len(y)
    if n <= max_points:
        return y
    binsize = int(math.ceil(n / float(max_points)))
    n_out = int(math.ceil(n / float(binsize)))
    mins = np.empty(n_out)
    maxs = np.empty(n_out)
    for i in range(n_out):
        s = i * binsize
        e = min(n, (i + 1) * binsize)
        seg = y[s:e]
        mins[i] = np.min(seg)
        maxs[i] = np.max(seg)
    y_ds = np.empty(n_out * 2)
    y_ds[0::2] = mins
    y_ds[1::2] = maxs
    return y_ds


# --- IO helper utilities usable without GUI ---
def save_model_obj(obj, path):
    """Save a model object with joblib or pickle."""
    try:
        if path.lower().endswith('.joblib') and joblib is not None:
            joblib.dump(obj, path); return
    except Exception:
        pass
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_model_obj(path):
    """Load a model object with joblib or pickle."""
    try:
        if path.lower().endswith('.joblib') and joblib is not None:
            return joblib.load(path)
    except Exception:
        pass
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_trials_csv(trials, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start_sample', 'end_sample'])
        for s, e in trials:
            w.writerow([int(s), int(e)])


def load_trials_csv(path):
    import csv
    out = []
    with open(path, 'r', newline='') as f:
        rdr = csv.reader(f)
        hdr = next(rdr, None)
        for row in rdr:
            if not row: continue
            try:
                s = int(row[0]); e = int(row[1])
            except Exception:
                continue
            out.append((s, e))
    return out


def save_features_npz(features, path):
    np.savez_compressed(path, features=features)


def load_features_npz(path):
    d = np.load(path, allow_pickle=True)
    if 'features' in d:
        return d['features']
    # try common keys
    for k in d.files:
        return d[k]
    return None

if _HAS_PYQT:
    class EMGViewerQt(QtWidgets.QMainWindow):
        """Minimal PyQt5-based EMG viewer to mirror Tk viewer functionality.

        This provides a lightweight, modern GUI using matplotlib's Qt backend.
        """
        def __init__(self, parent=None, num_channels=None, sample_rate=None):
            super().__init__(parent)
            self.setWindowTitle('EMG Viewer (PyQt)')
            self.data = None
            self.fs = float(sample_rate) if sample_rate is not None else 2000.0
            self.current_pos = 0
            self.playing = False
            self.max_points = 2000

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            vlay = QtWidgets.QVBoxLayout(central)

            # Create tabbed control area so we can port Tk tabs into PyQt
            tabs = QtWidgets.QTabWidget()

            # --- Acquisition tab (simple open/play controls) ---
            acq_tab = QtWidgets.QWidget()
            acq_layout = QtWidgets.QHBoxLayout(acq_tab)
            btn_open = QtWidgets.QPushButton('Open')
            btn_open.clicked.connect(self.open_file)
            acq_layout.addWidget(btn_open)
            self.btn_play = QtWidgets.QPushButton('Play')
            self.btn_play.clicked.connect(self._toggle_play)
            acq_layout.addWidget(self.btn_play)
            tabs.addTab(acq_tab, 'Acquisition')

            # --- Filtering tab ---
            filt_tab = QtWidgets.QWidget()
            filt_layout = QtWidgets.QHBoxLayout(filt_tab)
            filt_layout.addWidget(QtWidgets.QLabel('HP (Hz)'))
            self.spin_hp = QtWidgets.QDoubleSpinBox(); self.spin_hp.setRange(0.0, 1000.0); self.spin_hp.setValue(10.0)
            filt_layout.addWidget(self.spin_hp)
            filt_layout.addWidget(QtWidgets.QLabel('LP (Hz)'))
            self.spin_lp = QtWidgets.QDoubleSpinBox(); self.spin_lp.setRange(0.0, 10000.0); self.spin_lp.setValue(500.0)
            filt_layout.addWidget(self.spin_lp)
            filt_layout.addWidget(QtWidgets.QLabel('Notch'))
            self.combo_notch = QtWidgets.QComboBox(); self.combo_notch.addItems(['None', '50', '60'])
            filt_layout.addWidget(self.combo_notch)
            self.btn_apply_filters = QtWidgets.QPushButton('Apply Filters')
            self.btn_apply_filters.clicked.connect(self.apply_filters)
            filt_layout.addWidget(self.btn_apply_filters)
            filt_layout.addWidget(QtWidgets.QLabel('Channel'))
            self.chan_selector = QtWidgets.QSpinBox(); self.chan_selector.setRange(1,256); self.chan_selector.setValue(1)
            self.chan_selector.valueChanged.connect(self.plot_waveforms)
            filt_layout.addWidget(self.chan_selector)
            filt_layout.addWidget(QtWidgets.QLabel('FS'))
            self.spin_fs = QtWidgets.QSpinBox(); self.spin_fs.setRange(1,100000); self.spin_fs.setValue(int(self.fs))
            self.spin_fs.valueChanged.connect(self._on_fs_changed)
            filt_layout.addWidget(self.spin_fs)
            filt_layout.addWidget(QtWidgets.QLabel('Window (s)'))
            self.window_sec = QtWidgets.QDoubleSpinBox(); self.window_sec.setRange(0.01,60.0); self.window_sec.setValue(0.2)
            self.window_sec.valueChanged.connect(self.plot_waveforms)
            filt_layout.addWidget(self.window_sec)
            filt_layout.addWidget(QtWidgets.QLabel('Downsample'))
            self.spin_max_points = QtWidgets.QSpinBox(); self.spin_max_points.setRange(100,200000); self.spin_max_points.setValue(self.max_points)
            self.spin_max_points.valueChanged.connect(self._on_max_points_changed)
            filt_layout.addWidget(self.spin_max_points)
            filt_layout.addWidget(QtWidgets.QLabel('RMS win (s)'))
            self.rms_window_sec = QtWidgets.QDoubleSpinBox(); self.rms_window_sec.setRange(0.01,10.0); self.rms_window_sec.setSingleStep(0.01); self.rms_window_sec.setValue(0.05)
            filt_layout.addWidget(self.rms_window_sec)
            self.heat_live_checkbox = QtWidgets.QCheckBox('Heat Live'); self.heat_live_checkbox.setChecked(True)
            filt_layout.addWidget(self.heat_live_checkbox)
            tabs.addTab(filt_tab, 'Filtering')

            # --- Trials tab (placeholder, will implement segmentation/visualization) ---
            trials_tab = QtWidgets.QWidget()
            trials_layout = QtWidgets.QVBoxLayout(trials_tab)
            h = QtWidgets.QHBoxLayout()
            btn_detect = QtWidgets.QPushButton('Detect Trials')
            btn_detect.clicked.connect(self.detect_trials)
            h.addWidget(btn_detect)
            btn_load_events = QtWidgets.QPushButton('Load Events')
            btn_load_events.clicked.connect(self.load_events)
            h.addWidget(btn_load_events)
            trials_layout.addLayout(h)
            tabs.addTab(trials_tab, 'Trials')

            # --- Feature dataset tab (placeholder) ---
            feat_tab = QtWidgets.QWidget()
            feat_layout = QtWidgets.QHBoxLayout(feat_tab)
            btn_extract = QtWidgets.QPushButton('Extract Features')
            btn_extract.clicked.connect(self.extract_features_action)
            feat_layout.addWidget(btn_extract)
            btn_rms = QtWidgets.QPushButton('Compute RMS')
            btn_rms.clicked.connect(self.compute_rms_action)
            feat_layout.addWidget(btn_rms)
            btn_spec = QtWidgets.QPushButton('Show Spectrogram')
            btn_spec.clicked.connect(self.show_spectrogram_action)
            feat_layout.addWidget(btn_spec)
            btn_save_feat = QtWidgets.QPushButton('Save Features')
            btn_save_feat.clicked.connect(self.save_features)
            feat_layout.addWidget(btn_save_feat)
            tabs.addTab(feat_tab, 'Feature Dataset')

            vlay.addWidget(tabs)

            # Make waveform view taller by increasing figure height and stretch
            # plot area: waveform and RMS tabs
            self.plot_tabs = QtWidgets.QTabWidget()

            # waveform tab
            wave_w = QtWidgets.QWidget()
            wave_layout = QtWidgets.QVBoxLayout(wave_w)
            self.fig = MplFigure(figsize=(10, 4))
            self.canvas = FigureCanvasQt(self.fig)
            wave_layout.addWidget(self.canvas, 1)
            self.plot_tabs.addTab(wave_w, 'Waveform')

            # RMS heatmap tab
            heat_w = QtWidgets.QWidget()
            heat_layout = QtWidgets.QVBoxLayout(heat_w)
            self.fig_rms = MplFigure(figsize=(10, 4))
            self.canvas_rms = FigureCanvasQt(self.fig_rms)
            heat_layout.addWidget(self.canvas_rms, 1)
            self.plot_tabs.addTab(heat_w, 'RMS Heatmap')

            vlay.addWidget(self.plot_tabs, 3)

            self.scrub_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.scrub_slider.setMinimum(0); self.scrub_slider.setMaximum(0)
            self.scrub_slider.valueChanged.connect(self._on_slider_moved)
            try:
                self.scrub_slider.sliderPressed.connect(self._on_scrub_pressed)
                self.scrub_slider.sliderReleased.connect(self._on_scrub_released)
            except Exception:
                pass
            vlay.addWidget(self.scrub_slider)

            self.play_timer = QtCore.QTimer(); self.play_timer.setInterval(50); self.play_timer.timeout.connect(self._on_timer_tick)
            self._was_playing_on_scrub = False

            self.resize(1000,600)

        def _on_fs_changed(self, val): self.fs = float(val)
        def _on_max_points_changed(self, val): self.max_points = int(val); self.plot_waveforms()

        def open_file(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open EMG file', '', 'EMG Files (*.npz *.npy *.csv *.dat *.rhd);;RHD Files (*.rhd);;All Files (*)')
            if not path: return
            try:
                data = np.load(path) if path.lower().endswith('.npy') else np.loadtxt(path, delimiter=',')
            except Exception:
                try:
                    data = np.loadtxt(path)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Error', str(e))
                    return
            data = np.asarray(data)
            if data.ndim == 1: data = data[:,None]
            self.data = data
            self.chan_selector.setMaximum(max(1, self.data.shape[1]))
            self.scrub_slider.setMaximum(max(0, self.data.shape[0]-1))
            self.current_pos = 0
            self.plot_waveforms()

        def plot_waveforms(self):
            # draw on waveform figure/canvas
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            if self.data is None:
                ax.text(0.5,0.5,'Open a file to begin',ha='center',va='center')
                self.canvas.draw_idle(); return
            ch = int(self.chan_selector.value())-1
            fs = max(1.0, float(self.spin_fs.value()))
            window_s = float(self.window_sec.value())
            window_samples = max(1, int(window_s * fs))
            start = int(self.current_pos); end = min(self.data.shape[0], start + window_samples)
            t = np.arange(start, end) / fs
            y = self.data[start:end, ch]
            if len(y)==0:
                ax.text(0.5,0.5,'No data in window',ha='center',va='center'); self.canvas.draw_idle(); return
            # downsample by envelope
            ty = _downsample_for_plot(y, max_points=self.max_points)
            # If downsample returns an interleaved min/max envelope (length != t), build matching tx
            if ty is None:
                ax.plot(t, y, lw=0.8)
            elif len(ty) == len(t):
                ax.plot(t, ty, lw=0.8)
            else:
                # assume ty is interleaved min/max of length 2*n_out
                n_out = len(ty) // 2
                if n_out > 0:
                    binsize = int(np.ceil(len(y) / float(n_out)))
                    centers = np.empty(n_out)
                    for i in range(n_out):
                        s = i * binsize
                        e = min(len(y), (i+1)*binsize)
                        idx = s + (e - s) // 2
                        centers[i] = float((start + idx) / fs)
                    tx_ds = np.empty(n_out*2)
                    tx_ds[0::2] = centers
                    tx_ds[1::2] = centers
                    ax.plot(tx_ds, ty, lw=0.8)
                else:
                    ax.plot(t, y, lw=0.8)
            ax.set_xlabel('Time (s)'); ax.set_title(f'Channel {ch+1} [{start}:{end}]')
            self.fig.tight_layout(); self.canvas.draw_idle()

        def apply_filters(self):
            if self.data is None:
                QtWidgets.QMessageBox.warning(self, 'Filters', 'No data loaded')
                return
            hp = float(self.spin_hp.value())
            lp = float(self.spin_lp.value())
            notch_text = self.combo_notch.currentText()
            notch = None if notch_text == 'None' else float(notch_text)
            # Use scipy if available
            if butter is None or filtfilt is None:
                QtWidgets.QMessageBox.information(self, 'Filters', 'SciPy not available; skipping filter')
                return
            try:
                nyq = 0.5 * float(self.spin_fs.value())
                b, a = butter(4, [max(0.0001, hp/nyq), min(0.9999, lp/nyq)], btype='band')
                # apply along axis 0 (samples)
                self.data = filtfilt(b, a, self.data, axis=0)
                if notch is not None:
                    w0 = notch / nyq
                    bn, an = iirnotch(w0, 30)
                    self.data = filtfilt(bn, an, self.data, axis=0)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Filter error', str(e))
                return
            self.plot_waveforms()

        def plot_heatmap(self):
            # draw RMS heatmap into RMS figure/canvas
            if self.data is None:
                return
            rms_mean = np.sqrt(np.mean(self.data**2, axis=0))
            rows, cols = 8, 8
            grid = np.full((rows, cols), np.nan)
            nch = self.data.shape[1]
            for i in range(min(nch, rows*cols)):
                r = i // cols; c = i % cols; grid[r,c] = float(rms_mean[i])
            self.fig_rms.clear(); ax = self.fig_rms.add_subplot(111); im = ax.imshow(grid, cmap='inferno', aspect='auto')
            self.fig_rms.colorbar(im); self.canvas_rms.draw_idle()

        def _on_timer_tick(self):
            if self.data is None: return
            fs = float(max(1.0, float(self.spin_fs.value())))
            dt = self.play_timer.interval() / 1000.0
            step = max(1, int(fs * dt))
            self.current_pos = int(self.current_pos) + step
            window_s = float(self.window_sec.value())
            window_samples = max(1, int(window_s * fs))
            max_pos = max(0, self.data.shape[0] - window_samples)
            if self.current_pos >= max_pos:
                self.current_pos = max_pos
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')
            # update scrub slider without triggering extra updates
            try:
                self.scrub_slider.blockSignals(True)
                self.scrub_slider.setValue(int(self.current_pos))
            finally:
                try: self.scrub_slider.blockSignals(False)
                except Exception: pass
            self.plot_waveforms()

        def _toggle_play(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Play', 'No data loaded')
                return
            if self.playing:
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')
            else:
                # ensure slider maximum is set
                try:
                    self.scrub_slider.setMaximum(max(0, self.data.shape[0]-1))
                except Exception:
                    pass
                self.play_timer.start(); self.playing = True; self.btn_play.setText('Pause')

        def _on_slider_moved(self, value):
            self.current_pos = int(value)
            self.plot_waveforms()

        def _on_scrub_pressed(self):
            # pause playback while user scrubs
            self._was_playing_on_scrub = self.playing
            if self.playing:
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')

        def _on_scrub_released(self):
            # resume if it was playing
            if getattr(self, '_was_playing_on_scrub', False):
                self.play_timer.start(); self.playing = True; self.btn_play.setText('Pause')

        # --- Placeholder actions for tabs ported from Tk viewer ---
        def detect_trials(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Detect Trials', 'No data loaded')
                return
            ch = int(getattr(self, 'chan_selector', QtWidgets.QSpinBox()).value()) - 1
            if ch < 0 or ch >= self.data.shape[1]:
                QtWidgets.QMessageBox.information(self, 'Detect Trials', 'Invalid channel selected')
                return
            sig = self.data[:, ch].astype(float)
            fs = float(getattr(self, 'spin_fs', QtWidgets.QSpinBox()).value()) if hasattr(self, 'spin_fs') else self.fs
            win_s = float(getattr(self, 'rms_window_sec', QtWidgets.QDoubleSpinBox()).value()) if hasattr(self, 'rms_window_sec') else 0.05
            win = max(1, int(win_s * fs))
            # moving RMS
            try:
                kernel = np.ones(win) / float(win)
                rms = np.sqrt(np.convolve(sig**2, kernel, mode='same'))
            except Exception:
                rms = np.abs(sig)
            thresh = float(np.mean(rms) + 2.0 * np.std(rms))
            mask = rms > thresh
            # detect rising/falling edges
            d = np.diff(mask.astype(int))
            starts = list(np.where(d == 1)[0] + 1)
            ends = list(np.where(d == -1)[0] + 1)
            # handle edge cases
            if mask[0]:
                starts.insert(0, 0)
            if mask[-1]:
                ends.append(len(mask))
            trials = []
            for s, e in zip(starts, ends):
                trials.append((int(s), int(e)))
            self.trials = trials
            # show dialog with table of trials
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Detected Trials')
            dl = QtWidgets.QVBoxLayout(dlg)
            tbl = QtWidgets.QTableWidget()
            tbl.setColumnCount(3)
            tbl.setHorizontalHeaderLabels(['Index', 'Start (s)', 'End (s)'])
            tbl.setRowCount(len(trials))
            for i, (s, e) in enumerate(trials):
                tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
                tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{s/float(fs):.3f}"))
                tbl.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{e/float(fs):.3f}"))
            tbl.cellDoubleClicked.connect(lambda r, c: self._jump_to_trial_row(r, dlg))
            dl.addWidget(tbl)
            btns = QtWidgets.QHBoxLayout()
            save_btn = QtWidgets.QPushButton('Save CSV')
            def _save():
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Trials', '', 'CSV Files (*.csv);;All Files (*)')
                if not path:
                    return
                try:
                    import csv
                    with open(path, 'w', newline='') as f:
                        w = csv.writer(f)
                        w.writerow(['start_sample', 'end_sample'])
                        for s, e in trials:
                            w.writerow([s, e])
                    QtWidgets.QMessageBox.information(self, 'Save Trials', f'Saved {len(trials)} trials to {path}')
                except Exception as ex:
                    QtWidgets.QMessageBox.warning(self, 'Save Trials', str(ex))
            save_btn.clicked.connect(_save)
            btns.addWidget(save_btn)
            close_btn = QtWidgets.QPushButton('Close')
            close_btn.clicked.connect(dlg.accept)
            btns.addWidget(close_btn)
            dl.addLayout(btns)
            dlg.resize(400, 300)
            dlg.exec_()

        def load_events(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Events', '', 'CSV Files (*.csv);;All Files (*)')
            if not path:
                return
            try:
                import csv
                ev = []
                with open(path, 'r', newline='') as f:
                    rdr = csv.reader(f)
                    for row in rdr:
                        ev.append(row)
                self.events = ev
                QtWidgets.QMessageBox.information(self, 'Events', f'Loaded {len(ev)} event rows')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Events', str(e))

        def _jump_to_trial_row(self, row, dlg=None):
            if not hasattr(self, 'trials') or row < 0 or row >= len(self.trials):
                return
            s, e = self.trials[row]
            self.current_pos = int(s)
            try:
                self.scrub_slider.setValue(int(s))
            except Exception:
                pass
            self.plot_waveforms()
            if dlg is not None:
                dlg.accept()

        def load_trials(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Trials', '', 'CSV Files (*.csv);;All Files (*)')
            if not path:
                return
            try:
                import csv
                trials = []
                with open(path, 'r', newline='') as f:
                    rdr = csv.reader(f)
                    hdr = next(rdr, None)
                    for row in rdr:
                        if not row: continue
                        try:
                            s = int(row[0]); e = int(row[1])
                        except Exception:
                            continue
                        trials.append((s, e))
                self.trials = trials
                QtWidgets.QMessageBox.information(self, 'Load Trials', f'Loaded {len(trials)} trials')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Trials', str(e))

        def save_model(self):
            # Save the last trained model if present
            if not hasattr(self, 'trained_model') or self.trained_model is None:
                QtWidgets.QMessageBox.information(self, 'Save Model', 'No trained model to save')
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Model', '', 'Joblib (*.joblib);;Pickle (*.pkl);;All Files (*)')
            if not path:
                return
            try:
                if path.lower().endswith('.joblib') and joblib is not None:
                    joblib.dump(self.trained_model, path)
                else:
                    import pickle
                    with open(path, 'wb') as f:
                        pickle.dump(self.trained_model, f)
                QtWidgets.QMessageBox.information(self, 'Save Model', f'Model saved to {path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Save Model', str(e))

        def load_model(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Model', '', 'Joblib (*.joblib);;Pickle (*.pkl);;All Files (*)')
            if not path:
                return
            try:
                if path.lower().endswith('.joblib') and joblib is not None:
                    model = joblib.load(path)
                else:
                    import pickle
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                self.trained_model = model
                QtWidgets.QMessageBox.information(self, 'Load Model', f'Loaded model from {path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Model', str(e))

        def extract_features_action(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Features', 'No data loaded')
                return
            if extract_features is None:
                QtWidgets.QMessageBox.information(self, 'Features', 'Feature extraction function not available')
                return
            try:
                # expect data shape (samples, channels) -> extract_features handles shapes internally
                feats = extract_features(self.data, fs=self.fs)
                self.last_features = feats
                QtWidgets.QMessageBox.information(self, 'Features', f'Extracted features shape: {getattr(feats, "shape", "unknown")}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Features', str(e))

        def save_features(self):
            if not hasattr(self, 'last_features'):
                QtWidgets.QMessageBox.information(self, 'Save Features', 'No features to save')
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Features', '', 'NumPy (.npz);;All Files (*)')
            if not path:
                return
            try:
                np.savez_compressed(path, features=self.last_features)
                QtWidgets.QMessageBox.information(self, 'Save Features', f'Saved features to {path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Save Features', str(e))

        def compute_rms_action(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'RMS', 'No data loaded')
                return
            ch = int(getattr(self, 'chan_selector', QtWidgets.QSpinBox()).value()) - 1
            if ch < 0 or ch >= self.data.shape[1]:
                QtWidgets.QMessageBox.information(self, 'RMS', 'Invalid channel selected')
                return
            sig = self.data[:, ch].astype(float)
            fs = float(getattr(self, 'spin_fs', QtWidgets.QSpinBox()).value()) if hasattr(self, 'spin_fs') else self.fs
            win_s = float(getattr(self, 'rms_window_sec', QtWidgets.QDoubleSpinBox()).value()) if hasattr(self, 'rms_window_sec') else 0.05
            win = max(1, int(win_s * fs))
            kernel = np.ones(win) / float(win)
            try:
                rms = np.sqrt(np.convolve(sig**2, kernel, mode='same'))
            except Exception:
                rms = np.abs(sig)
            self.last_rms = rms
            t = np.arange(len(rms)) / float(fs)
            self.fig.clear(); ax = self.fig.add_subplot(111)
            ax.plot(t, rms, lw=0.8)
            ax.set_title(f'RMS (ch {ch+1})')
            ax.set_xlabel('Time (s)')
            self.fig.tight_layout(); self.canvas.draw_idle()

        def show_spectrogram_action(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Spectrogram', 'No data loaded')
                return
            if spectrogram is None:
                QtWidgets.QMessageBox.information(self, 'Spectrogram', 'SciPy not available; cannot compute spectrogram')
                return
            ch = int(getattr(self, 'chan_selector', QtWidgets.QSpinBox()).value()) - 1
            sig = self.data[:, ch].astype(float)
            fs = float(getattr(self, 'spin_fs', QtWidgets.QSpinBox()).value()) if hasattr(self, 'spin_fs') else self.fs
            try:
                f, t, Sxx = spectrogram(sig, fs=fs, nperseg=256, noverlap=128)
                self.fig.clear(); ax = self.fig.add_subplot(111)
                im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='auto')
                ax.set_ylabel('Frequency [Hz]'); ax.set_xlabel('Time [s]')
                self.fig.colorbar(im)
                self.fig.tight_layout(); self.canvas.draw_idle()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Spectrogram', str(e))

# TO-DO: implement this into a separate intan module
try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    _HAS_SKLEARN = True
except Exception:
    PCA = None
    LabelEncoder = None
    train_test_split = None
    _HAS_SKLEARN = False

class EMGViewerTk:
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Data Viewer")

        # === Initialize variables ===
        # EMG data (channels x samples)
        self.emg_data_raw = None
        self.segment_data = None
        self.segment_fs = None
        self.sampling_rate = None
        self.time_vector = None
        self.current_channel = 0

        # UI control variables
        self.domain_mode = tk.StringVar(value="Time")
        self.notch_enabled = tk.BooleanVar(value=True)
        self.show_lines = tk.BooleanVar(value=True)
        self.car_enabled = tk.BooleanVar(value=False)
        self.xlim = None
        self.ylim = None
        #self.data_viewer_file_path = None

        # Filtering tab entries
        self.low_cut = tk.StringVar(value="20")
        self.high_cut = tk.StringVar(value="450")
        self.filter_order = tk.StringVar(value="4")

        # Training/feature extraction controls (example)
        self.training_features = {
            "bandpass": tk.BooleanVar(value=True),
            "notch": tk.BooleanVar(value=True),
            "rectify": tk.BooleanVar(value=False),
            "envelop_smooth": tk.BooleanVar(value=False),
            "car": tk.BooleanVar(value=False),
            # Feature toggles:
            "mean_absolute_value": tk.BooleanVar(value=True),
            "zero_crossings": tk.BooleanVar(value=False),
            "slope_sign_changes": tk.BooleanVar(value=False),
            "waveform_length": tk.BooleanVar(value=False),
            "rms": tk.BooleanVar(value=True),
        }

        self.feature_controls = {
            "zero_crossings": {"threshold": tk.StringVar(value="0.01")},
            "slope_sign_changes": {"delta_threshold": tk.StringVar(value="0.01")},
            "waveform_length": {"window_ms": tk.StringVar(value="200")},
        }
        self.smooth_f_entry = tk.StringVar(value="5.0")  # Envelope smoothing freq (Hz)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.build_layout()

    def build_layout(self):
        # === Top Frame for Tabs + Sidebar ===
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side="top", fill="both", expand=True)

        # === Tabs ===
        #self.tabs = ttk.Notebook(self.root)
        self.tabs = ttk.Notebook(top_frame)
        self.tabs.pack(side="left", fill="both", expand=True)

        self.tab_acquisition = ttk.Frame(self.tabs)
        self.tab_filtering = ttk.Frame(self.tabs)
        self.tab_trials = ttk.Frame(self.tabs)
        self.tab_training = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_acquisition, text="Data Acquisition")
        self.tabs.add(self.tab_filtering, text="Filtering")
        self.tabs.add(self.tab_trials, text="Trial Utilities")
        self.tabs.add(self.tab_training, text="Feature Dataset")

        # === Plotting Area ===
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side="bottom", fill="both", expand=True)

        plot_frame = ttk.Frame(bottom_frame)
        plot_frame.pack(side="left", fill="both", expand=True)

        # Plot canvas
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Toolbar (now visually docked into the plot area)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")

        self.scroll = tk.Scrollbar(bottom_frame, orient="horizontal", command=self.scroll_plot)
        self.scroll.pack(side="bottom", fill="x")

        # === Sidebar for Channel Controls
        side_controls = ttk.Frame(bottom_frame)
        side_controls.pack(side="right", fill="y", padx=10)

        ttk.Label(side_controls, text="Channel:").pack(anchor="w")
        self.channel_selector = ttk.Combobox(side_controls, state="readonly", width=10)
        self.channel_selector.bind("<<ComboboxSelected>>", self.update_channel)
        self.channel_selector.pack(fill="x", pady=5)

        ttk.Label(side_controls, text="Domain:").pack(anchor="w")
        self.domain_dropdown = ttk.Combobox(side_controls, state="readonly", textvariable=self.domain_mode,
                                            values=["Time", "PSD", "Spectrogram", "Waterfall", "Features"], width=10)
        self.domain_dropdown.bind("<<ComboboxSelected>>", self.plot_channel)
        self.domain_dropdown.pack(pady=5)

        # Create the individual tabs
        self.build_acquisition_tab()
        self.build_filtering_tab()
        self.build_trials_tab()
        self.build_training_tab()

    def build_acquisition_tab(self):
        frame = ttk.Frame(self.tab_acquisition)
        frame.pack(fill="x", pady=5)

        ttk.Button(frame, text="Load File", command=self.load_file).pack(side="left", padx=5)

        control_frame = ttk.Frame(self.tab_acquisition)
        control_frame.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Waterfall Channels:").pack(anchor="w")
        self.channel_range_entry = ttk.Entry(control_frame, width=15)
        self.channel_range_entry.insert(0, "0-15")  # default view
        self.channel_range_entry.pack(pady=5)

    def build_trials_tab(self):

        # === Left controls ====
        container = ttk.Frame(self.tab_trials)
        container.pack(fill="both", expand=True)

        left_panel = ttk.Frame(container)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Label(left_panel, text="Custom Label:").pack(anchor="w")
        self.label_entry = ttk.Entry(left_panel)
        self.label_entry.insert(0, "Label")
        self.label_entry.pack(fill="x", pady=5)

        ttk.Checkbutton(left_panel, text="Show Trial Markers", variable=self.show_lines,
                        command=self.plot_channel).pack(anchor="w", pady=5)
        ttk.Button(left_panel, text="Enable Indexing", command=self.enable_indexing).pack(anchor="w", pady=5)

        # === Right side: trial label table ===
        right_panel = ttk.Frame(container)
        right_panel.pack(side="right", fill="both", expand=True)

        self.table = ttk.Treeview(right_panel, columns=("Sample Index", "Label"), show="headings", height=10)
        self.table.heading("Sample Index", text="Sample Index")
        self.table.heading("Label", text="Label")
        self.table.pack(fill="x", pady=5, expand=True)

        button_frame = ttk.Frame(right_panel)
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="Save", command=self.save_table).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load", command=self.load_table).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_selected).pack(side="left", padx=5)

        ttk.Button(right_panel, text="Run Trial Segmentation", command=self.run_trial_segmentation).pack(pady=5)
        ttk.Button(right_panel, text="Build Training Set", command=self.build_training_dataset).pack(pady=5)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.indexing_enabled = False

    def build_filtering_tab(self):

        # === Top-level horizontal frame for grouping ===
        settings_container = ttk.Frame(self.tab_filtering)
        settings_container.pack(fill="x", padx=5, pady=5)

        # ===== Filter settings ===========
        control_frame = ttk.LabelFrame(settings_container, text="Filter Settings")
        control_frame.pack(side="left", padx=5, pady=5, anchor="n")

        ttk.Label(control_frame, text="Filter Type:").grid(row=0, column=0)
        self.filter_type = ttk.Combobox(control_frame, values=["bandpass"], state="readonly")
        self.filter_type.current(0)
        self.filter_type.grid(row=0, column=1)

        ttk.Label(control_frame, text="Low Cut (Hz):").grid(row=1, column=0)
        self.low_cut = ttk.Entry(control_frame)
        self.low_cut.insert(0, "120")
        self.low_cut.grid(row=1, column=1)
        self.low_cut.bind("<Return>", lambda e: self.plot_channel())
        self.low_cut.bind("<FocusOut>", lambda e: self.plot_channel())

        ttk.Label(control_frame, text="High Cut (Hz):").grid(row=2, column=0)
        self.high_cut = ttk.Entry(control_frame)
        self.high_cut.insert(0, "1000")
        self.high_cut.grid(row=2, column=1)
        self.high_cut.bind("<Return>", lambda e: self.plot_channel())
        self.high_cut.bind("<FocusOut>", lambda e: self.plot_channel())

        ttk.Label(control_frame, text="Order:").grid(row=3, column=0)
        self.filter_order = ttk.Entry(control_frame)
        self.filter_order.insert(0, "4")
        self.filter_order.grid(row=3, column=1)
        self.filter_order.bind("<Return>", lambda e: self.plot_channel())
        self.filter_order.bind("<FocusOut>", lambda e: self.plot_channel())

        self.notch_check = ttk.Checkbutton(control_frame, text="60Hz Notch Filter", variable=self.notch_enabled)
        self.notch_check.grid(row=4, column=0, columnspan=2)
        self.notch_check = ttk.Checkbutton(control_frame, text="60Hz Notch Filter", variable=self.notch_enabled,
                                           command=self.plot_channel)

        # === CAR CHECKBOX ===
        self.car_check = ttk.Checkbutton(control_frame, text="Enable Common Avg Reference", variable=self.car_enabled,
                                         command=self.plot_channel)
        self.car_check.grid(row=5, column=0, columnspan=2, sticky='w')

        #ttk.Button(control_frame, text="Apply Filter", command=self.plot_channel()).grid(row=5, column=0, columnspan=2, pady=5)

        # ===== PSD Settings ===========
        psd_frame = ttk.LabelFrame(settings_container, text="PSD Settings")
        psd_frame.pack(side="left", padx=5, pady=5, anchor="n")

        ttk.Label(psd_frame, text="NperSeg:").grid(row=0, column=0)
        self.nperseg_entry = ttk.Entry(psd_frame)
        self.nperseg_entry.insert(0, "256")
        self.nperseg_entry.grid(row=0, column=1)
        self.nperseg_entry.bind("<FocusOut>", lambda e: self.plot_channel())
        self.nperseg_entry.bind("<Return>", lambda e: self.plot_channel())

        ttk.Label(psd_frame, text="Min Freq (Hz):").grid(row=0, column=0)
        self.psd_min_freq = ttk.Entry(psd_frame)
        self.psd_min_freq.insert(0, "0")
        self.psd_min_freq.grid(row=0, column=1)
        self.psd_min_freq.bind("<FocusOut>", lambda e: self.plot_channel())
        self.psd_min_freq.bind("<Return>", lambda e: self.plot_channel())

        ttk.Label(psd_frame, text="Max Freq (Hz):").grid(row=1, column=0)
        self.psd_max_freq = ttk.Entry(psd_frame)
        self.psd_max_freq.insert(0, "1000")
        self.psd_max_freq.grid(row=1, column=1)
        self.psd_max_freq.bind("<FocusOut>", lambda e: self.plot_channel())
        self.psd_max_freq.bind("<Return>", lambda e: self.plot_channel())


        # ======= Spectrogram Settings ===========
        spec_frame = ttk.LabelFrame(settings_container, text="Spectrogram Settings")
        spec_frame.pack(side="left", padx=5, pady=5, anchor="n")

        ttk.Label(spec_frame, text="NFFT:").grid(row=0, column=0)
        self.nfft_entry = ttk.Entry(spec_frame)
        self.nfft_entry.insert(0, "256")
        self.nfft_entry.grid(row=0, column=1)
        self.nfft_entry.bind("<FocusOut>", lambda e: self.plot_channel())
        self.nfft_entry.bind("<Return>", lambda e: self.plot_channel())

        ttk.Label(spec_frame, text="No. Overlap:").grid(row=1, column=0)
        self.noverlap_entry = ttk.Entry(spec_frame)
        self.noverlap_entry.insert(0, "128")
        self.noverlap_entry.grid(row=1, column=1)
        self.noverlap_entry.bind("<FocusOut>", lambda e: self.plot_channel())
        self.noverlap_entry.bind("<Return>", lambda e: self.plot_channel())

        self.cmap_entry = ttk.Entry(spec_frame)
        self.cmap_entry.insert(0, "viridis")
        ttk.Label(spec_frame, text="Colormap:").grid(row=6, column=0)
        self.cmap_entry.grid(row=6, column=1)
        self.cmap_entry.bind("<FocusOut>", lambda e: self.plot_channel())
        self.cmap_entry.bind("<Return>", lambda e: self.plot_channel())

        # Set the frequency range limits to view as min/max
        self.spec_freq_min = ttk.Entry(spec_frame)
        self.spec_freq_min.insert(0, "0")
        ttk.Label(spec_frame, text="Freq Min (Hz):").grid(row=7, column=0)
        self.spec_freq_min.grid(row=7, column=1)
        self.spec_freq_min.bind("<FocusOut>", lambda e: self.plot_channel())
        self.spec_freq_min.bind("<Return>", lambda e: self.plot_channel())

        self.spec_freq_max = ttk.Entry(spec_frame)
        self.spec_freq_max.insert(0, "1000")
        ttk.Label(spec_frame, text="Freq Max (Hz):").grid(row=8, column=0)
        self.spec_freq_max.grid(row=8, column=1)
        self.spec_freq_max.bind("<FocusOut>", lambda e: self.plot_channel())
        self.spec_freq_max.bind("<Return>", lambda e: self.plot_channel())

        # Time Range: # to #
        self.spec_time_range_min = ttk.Entry(spec_frame)
        self.spec_time_range_min.insert(0, "0")
        ttk.Label(spec_frame, text="Time Min (s):").grid(row=9, column=0)
        self.spec_time_range_min.grid(row=9, column=1)
        self.spec_time_range_min.bind("<FocusOut>", lambda e: self.plot_channel())
        self.spec_time_range_min.bind("<Return>", lambda e: self.plot_channel())

        self.spec_time_range_max = ttk.Entry(spec_frame)
        self.spec_time_range_max.insert(0, "10")
        ttk.Label(spec_frame, text="Time Max (s):").grid(row=10, column=0)
        self.spec_time_range_max.grid(row=10, column=1)
        self.spec_time_range_max.bind("<FocusOut>", lambda e: self.plot_channel())
        self.spec_time_range_max.bind("<Return>", lambda e: self.plot_channel())

    def build_training_tab(self):
        container = ttk.Frame(self.tab_training)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # === Left: Directory Table ===
        left_panel = ttk.Frame(container)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Scrollable Treeview Frame
        treeview_frame = ttk.Frame(left_panel)
        treeview_frame.pack(fill="both", expand=True)

        # Treeview
        self.training_dir_table = ttk.Treeview(treeview_frame, columns=["Path"], show="headings", height=10)
        self.training_dir_table.heading("Path", text="Path")
        self.training_dir_table.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(treeview_frame, orient="vertical", command=self.training_dir_table.yview)
        scrollbar.pack(side="right", fill="y")
        self.training_dir_table.configure(yscrollcommand=scrollbar.set)

        button_frame = ttk.Frame(left_panel)
        button_frame.pack(pady=5, fill="x")
        ttk.Button(button_frame, text="Add Directory", command=self.add_training_directory).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Remove Directory", command=self.remove_training_directory).pack(side="left")
        ttk.Button(button_frame, text="Load Segment", command=self.load_segment_and_visualize).pack(side="left",
                                                                                                    padx=(5, 0))

        # === Middle: Label Viewer ===
        middle_panel = ttk.Frame(container)
        middle_panel.pack(side="left", fill="y", padx=(5, 10))

        ttk.Label(middle_panel, text="Detected Labels:").pack(anchor="w")
        self.training_labels_listbox = tk.Listbox(middle_panel, height=20, exportselection=False)
        self.training_labels_listbox.pack(pady=5, expand=True)

        label_button_frame = ttk.Frame(middle_panel)
        label_button_frame.pack(pady=5, fill="x")
        ttk.Button(label_button_frame, text="Refresh Labels", command=self.refresh_labels_list).pack(side="left",
                                                                                                     padx=(0, 5))
        ttk.Button(label_button_frame, text="Remove Label", command=self.remove_selected_label).pack(side="left")

        # === Right: Feature + Preprocessing Settings ===
        right_panel = ttk.Frame(container)
        right_panel.pack(side="left", fill="y")

        # === Preprocessing Settings ===
        pre_frame = ttk.LabelFrame(right_panel, text="Preprocessing", padding=(5, 5))
        pre_frame.pack(fill="x", pady=(0, 10))

        self.training_features = {}
        self.feature_controls = {}

        ttk.Button(pre_frame, text="Copy Display Filter Settings", command=self.copy_display_to_training_filters).pack(
            anchor="w")

        self.training_features["notch"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(pre_frame, text="Notch Filter (60 Hz)", variable=self.training_features["notch"]).pack(
            anchor="w")

        self.training_features["bandpass"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(pre_frame, text="Bandpass Filter", variable=self.training_features["bandpass"]).pack(anchor="w")

        self.training_features["rectify"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(pre_frame, text="Rectify Signal", variable=self.training_features["rectify"]).pack(anchor="w")

        self.training_features["envelop_smooth"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(pre_frame, text="Envelope Smooth", variable=self.training_features["envelop_smooth"]).pack(
            anchor="w")

        ttk.Label(pre_frame, text="Envelope LF (Hz):").pack(anchor="w")
        self.smooth_f_entry = ttk.Entry(pre_frame, width=10)
        self.smooth_f_entry.insert(0, "5")
        self.smooth_f_entry.pack(anchor="w", pady=(0, 5))

        # === Feature Extraction Settings ===
        feat_frame = ttk.LabelFrame(right_panel, text="Feature Extraction", padding=(5, 5))
        feat_frame.pack(fill="x", pady=(0, 10))

        self.training_features["use_sliding_window"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(feat_frame, text="Use Sliding Window",
                        variable=self.training_features["use_sliding_window"]).pack(anchor="w")

        ttk.Label(feat_frame, text="Window Size (ms):").pack(anchor="w")
        self.window_size_entry = ttk.Entry(feat_frame, width=10)
        self.window_size_entry.insert(0, "250")
        self.window_size_entry.pack(anchor="w")
        ttk.Label(feat_frame, text="Step Size (ms):").pack(anchor="w")
        self.step_size_entry = ttk.Entry(feat_frame, width=10)
        self.step_size_entry.insert(0, "50")
        self.step_size_entry.pack(anchor="w")

        self.training_features["rms"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(feat_frame, text="RMS", variable=self.training_features["rms"]).pack(anchor="w")

        self.training_features["mean_absolute_value"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(feat_frame, text="Mean Absolute Value",
                        variable=self.training_features["mean_absolute_value"]).pack(anchor="w")

        self.training_features["zero_crossings"] = tk.BooleanVar(value=True)
        frame_zc = ttk.Frame(feat_frame)
        frame_zc.pack(anchor="w", fill="x")
        ttk.Checkbutton(frame_zc, text="Zero Crossings", variable=self.training_features["zero_crossings"]).pack(
            side="left")
        ttk.Label(frame_zc, text="Thresh:").pack(side="left")
        zc_thresh = ttk.Entry(frame_zc, width=6)
        zc_thresh.insert(0, "0.01")
        zc_thresh.pack(side="left")
        self.feature_controls["zero_crossings"] = {"threshold": zc_thresh}

        self.training_features["slope_sign_changes"] = tk.BooleanVar(value=True)
        frame_ssc = ttk.Frame(feat_frame)
        frame_ssc.pack(anchor="w", fill="x")
        ttk.Checkbutton(frame_ssc, text="Slope Sign Changes",
                        variable=self.training_features["slope_sign_changes"]).pack(side="left")
        ttk.Label(frame_ssc, text="ΔThresh:").pack(side="left")
        ssc_thresh = ttk.Entry(frame_ssc, width=6)
        ssc_thresh.insert(0, "0.01")
        ssc_thresh.pack(side="left")
        self.feature_controls["slope_sign_changes"] = {"delta_threshold": ssc_thresh}

        self.training_features["waveform_length"] = tk.BooleanVar(value=True)
        frame_wl = ttk.Frame(feat_frame)
        frame_wl.pack(anchor="w", fill="x")
        ttk.Checkbutton(frame_wl, text="Waveform Length", variable=self.training_features["waveform_length"]).pack(
            side="left")
        ttk.Label(frame_wl, text="Window (ms):").pack(side="left")
        wl_window_entry = ttk.Entry(frame_wl, width=6)
        wl_window_entry.insert(0, "200")
        wl_window_entry.pack(side="left")
        self.feature_controls["waveform_length"] = {"window_ms": wl_window_entry}

        # === PCA Components ===
        ttk.Label(right_panel, text="PCA Components:").pack(anchor="w", pady=(10, 0))
        self.pca_components_entry = ttk.Entry(right_panel, width=10)
        self.pca_components_entry.insert(0, "50")
        self.pca_components_entry.pack(anchor="w", pady=(0, 10))

        ttk.Button(right_panel, text="Build Training Set", command=self.build_training_dataset).pack(pady=15)

    def add_feature_directory(self):
        path = filedialog.askdirectory(title="Select EMG Segment Directory")
        if path:
            self.features_dir_listbox.insert("end", path)

    def add_training_directory(self):
        folder_path = filedialog.askdirectory(title="Select EMG Segment Directory")
        if not folder_path:
            return

        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".npz"):
                full_path = os.path.join(folder_path, fname)
                self.training_dir_table.insert("", "end", values=[full_path])
                self.update_training_labels_from_filename(fname)

    def apply_visualization_filters(self, signal):
        """
        Filtering for visualization using Filtering Tab controls.
        """
        x = np.atleast_2d(signal)
        if x.shape[0] > x.shape[1]:
            x = x.T  # (channels, samples)
        fs = self.sampling_rate or 2000

        # CAR (display)
        if self.car_enabled.get() and x.shape[0] > 1:
            x = common_average_reference(x)

        # Bandpass (display)
        try:
            low = float(self.low_cut.get())
            high = float(self.high_cut.get())
            order = int(self.filter_order.get())
            if fs/2 > high > low > 0:
                x = bandpass_filter(x, lowcut=low, highcut=high, fs=fs, order=order, axis=1)
        except Exception as e:
            print(f"[Visualization bandpass error] {e}")

        # Notch (display)
        if hasattr(self, 'notch_enabled') and self.notch_enabled.get():
            try:
                x = notch_filter(x, fs=fs, f0=60, Q=30, axis=1)
            except Exception as e:
                print(f"[Visualization notch error] {e}")

        return np.squeeze(x)

    def copy_display_to_training_filters(self):
        # If you have dedicated bandpass entries for training filters
        self.bp_low_entry.delete(0, 'end')
        self.bp_low_entry.insert(0, self.low_cut.get())
        self.bp_high_entry.delete(0, 'end')
        self.bp_high_entry.insert(0, self.high_cut.get())
        self.bp_order_entry.delete(0, 'end')
        self.bp_order_entry.insert(0, self.filter_order.get())
        self.training_features["notch"].set(self.notch_enabled.get())
        self.training_features["car"].set(self.car_enabled.get())

    def load_feature_segments(self):
        self.features_segment_listbox.delete(0, "end")
        for i in range(self.features_dir_listbox.size()):
            dir_path = self.features_dir_listbox.get(i)
            if os.path.exists(dir_path):
                for fname in sorted(os.listdir(dir_path)):
                    if fname.endswith(".npz"):
                        full_path = os.path.join(dir_path, fname)
                        self.features_segment_listbox.insert("end", full_path)

    def load_segment_and_visualize(self):
        selection = self.training_dir_table.selection()
        if not selection:
            print("No segment file selected.")
            return

        path = self.training_dir_table.item(selection[0])["values"][0]
        if not os.path.exists(path):
            print(f"Segment file not found: {path}")
            return

        # Load EMG data
        data = np.load(path, allow_pickle=True)
        emg = data["emg"]
        fs = float(data.get("fs", self.sampling_rate or 2000))

        # Store for later plotting
        emg = self.apply_training_filters(emg)

        self.segment_data = emg
        self.segment_fs = fs

        # Update channel selector to reflect segment shape
        self.channel_selector["values"] = [f"Ch {i}" for i in range(emg.shape[0])]
        self.channel_selector.current(0)
        self.current_channel = 0

        # Force plot update
        self.domain_mode.set("Features")
        self.plot_channel()
        print("Segment loaded and visualized.")

    def load_file(self):
        """ Load a .rhd file and plot the first channel."""
        path = filedialog.askopenfilename(filetypes=[
            ("RHD files", "*.rhd"),
            ("CSV Files", "*.csv"),
        ])
        if not path:
            return
        if path.endswith('.csv'):
            # Load CSV file. first column has timestamp data in milliseconds elapsed, the rest are EMG channels if
            # they have "EMG" in the name. The first row has only header information
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            self.time_vector = data[:, 0] / 1000.0

            # Find the columns that contain "EMG" in their header
            with open(path, 'r') as f:
                header = f.readline().strip().split(',')
            emg_columns = [i for i, col in enumerate(header) if "EMG" in col]
            self.emg_data = data[:, emg_columns].T

            self.sampling_rate = 1000.0 / (self.time_vector[1] - self.time_vector[0])  # Assuming uniform sampling
        elif path.endswith('.rhd') and load_rhd_file is not None:
            try:
                result = load_rhd_file(path)
                self.emg_data = result.get("amplifier_data") or result.get('data')
                # fallback keys
                self.time_vector = result.get("t_amplifier") or result.get('t')
                self.sampling_rate = result.get("frequency_parameters", {}).get("amplifier_sample_rate")
            except Exception as e:
                print(f"Failed to load .rhd with intan.io: {e}")
                return
        else:
            # fallback: try csv or numpy loader
            try:
                if path.endswith('.npy') or path.endswith('.npz'):
                    arr = np.load(path)
                    if isinstance(arr, np.lib.npyio.NpzFile):
                        self.emg_data = arr[arr.files[0]]
                    else:
                        self.emg_data = arr
                else:
                    data = np.loadtxt(path, delimiter=',')
                    # assume first column may be time
                    if data.ndim > 1 and data.shape[1] > 1:
                        self.time_vector = data[:, 0] / 1000.0
                        self.emg_data = data[:, 1:].T
                    else:
                        self.emg_data = data.T
            except Exception as e:
                print(f"Failed to load file: {e}")
                return

        self.data_viewer_file_path = path
        self.channel_selector["values"] = [f"Ch {i}" for i in range(self.emg_data.shape[0])]
        self.channel_selector.current(0)
        self.current_channel = 0
        self.plot_channel()

    def update_training_labels_from_directory(self):
        """ Helper function to refresh the training labels listbox with all the labels detected from the current directories"""
        # Delete all labels from the listbox
        self.training_labels_listbox.delete(0, "end")

        # Iterate through each directory in training_dir_table
        for item in self.training_dir_table.get_children():
            path = self.training_dir_table.item(item)["values"][0]
            if os.path.exists(path):
                self.update_training_labels_from_filename(path)

    def remove_training_directory(self):
        selected = self.training_dir_table.selection()
        for item in selected:
            self.training_dir_table.delete(item)
        self.refresh_labels_list()

    def remove_selected_label(self):
        selection = self.training_labels_listbox.curselection()
        if selection:
            self.training_labels_listbox.delete(selection[0])
        else:
            print("Please select a label to remove.")

    def visualize_selected_segment(self):
        selection = self.features_segment_listbox.curselection()
        if not selection:
            print("Please select a segment file to view.")
            return

        path = self.features_segment_listbox.get(selection[0])
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

        data = np.load(path, allow_pickle=True)
        emg = data["emg"]
        fs = data.get("fs", self.sampling_rate or 2000)

        # Apply preprocessing
        if self.features_filters["bandpass"].get():
            b, a = butter(4, [20, 450], btype="band", fs=fs)
            emg = filtfilt(b, a, emg, axis=1)
        if self.features_filters["notch"].get():
            b, a = iirnotch(60, 30, fs)
            emg = filtfilt(b, a, emg, axis=1)
        if self.features_filters["rectify"].get():
            emg = np.abs(emg)
        if self.features_filters["smooth_rms"].get():
            kernel = np.ones(int(0.2 * fs)) / int(0.2 * fs)
            emg = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=emg)

        self.ax.clear()

        # Plot preprocessed EMG (all channels with offset)
        offset = 0
        for i, ch in enumerate(emg):
            self.ax.plot(np.arange(len(ch)) / fs, ch + offset, label=f"Ch {i}", linewidth=0.6)
            offset += np.max(np.abs(ch)) * 2  # dynamic spacing

        self.ax.set_title("Segment with Preprocessing")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude + offset")
        self.ax.grid(True)

        self.canvas.draw()
        print("Segment visualization updated.")

    def run_pca_visualization(self):
        path = filedialog.askopenfilename(title="Select Feature CSV Dataset", filetypes=[("CSV Files", "*.csv")])
        if not path or not os.path.exists(path):
            print("No dataset selected or file does not exist.")
            return

        df = pd.read_csv(path)
        if "Label" not in df.columns:
            print("No 'Label' column found in dataset.")
            return

        features = df.drop(columns=["Label"]).values
        labels = df["Label"].values

        # Encode labels if necessary
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        # Apply PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)

        # Plot results
        self.ax.clear()
        for i, label in enumerate(np.unique(labels_encoded)):
            idx = labels_encoded == label
            self.ax.scatter(reduced[idx, 0], reduced[idx, 1], label=le.classes_[i], alpha=0.6)
        self.ax.set_title("2D PCA of EMG Features")
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.legend()
        self.canvas.draw()

    def refresh_labels_list(self):
        """
        Clear and rebuild the labels list from all currently listed directories.
        """
        self.training_labels_listbox.delete(0, "end")
        for item in self.training_dir_table.get_children():
            path = self.training_dir_table.item(item)["values"][0]
            self.update_training_labels_from_directory(path)

    def update_training_labels_from_filename(self, fname):
        """
        Extract label from filename and add to label list if not already present.
        Assumes filenames like: participant_label_0.npz
        """
        parts = fname.replace(".npz", "").split("_")
        if len(parts) >= 2:
            label = parts[-2].strip().lower()
            if label and label not in self.training_labels_listbox.get(0, "end"):
                self.training_labels_listbox.insert("end", label)

    def run_trial_segmentation(self):
        """
        Segment the EMG data into trials based on manual indices or notes files.
        Saves each segment as a compressed .npz file (with 'emg' data and 'label') in an 'emg' subfolder.
        """
        from intan.io import load_rhd_file, load_labeled_file
        # Determine output directory for segmented files
        if self.data_viewer_file_path:
            # If a file is currently loaded in the viewer, use its directory
            raw_folder = os.path.dirname(self.data_viewer_file_path)
        else:
            # Otherwise, ask the user to select the top-level raw data directory
            raw_folder = filedialog.askdirectory(title="Select Raw Data Directory")
            if not raw_folder:
                return  # user canceled

        out_dir = os.path.join(os.path.dirname(raw_folder), "emg")  # output subdirectory for segments
        os.makedirs(out_dir, exist_ok=True)

        # If manual markers exist in the table and an EMG file is loaded, segment using those
        if self.emg_data is not None and len(self.table.get_children()) > 0:
            # Get all markers from the table
            markers = []
            for row in self.table.get_children():
                sample_idx, label = self.table.item(row)["values"]
                markers.append((int(sample_idx), str(label)))
            # Sort markers by sample index
            markers.sort(key=lambda x: x[0])
            # Convert to DataFrame for consistency
            notes_df = pd.DataFrame(markers, columns=["Sample", "Label"])
            # Add a final marker at end of recording to segment the last trial (if not already present)
            total_samples = self.emg_data.shape[1]
            if notes_df.iloc[-1]["Sample"] != total_samples:
                notes_df = pd.concat([notes_df, pd.DataFrame([[total_samples, ""]], columns=["Sample", "Label"])],
                                     ignore_index=True)
            # Use the loaded EMG data and manual markers for segmentation
            trial_name = os.path.basename(raw_folder)  # use folder name as trial identifier
            for i in range(len(notes_df) - 1):
                label = notes_df.loc[i, "Label"]
                if str(label).strip() == "" or str(label).lower() == "nan":
                    continue  # skip empty labels (if any)
                start_idx = notes_df.loc[i, "Sample"]
                end_idx = notes_df.loc[i + 1, "Sample"]
                segment = self.emg_data[:, start_idx:end_idx]
                # Safe label for filename (no spaces, lowercase)
                label_safe = str(label).strip().replace(" ", "_").lower()
                file_name = f"{trial_name}_{label_safe}_{i}.npz"
                save_path = os.path.join(out_dir, file_name)
                # Save segment with label and include sampling rate for reference
                np.savez_compressed(save_path, emg=segment, label=label, fs=self.sampling_rate)
                print(f"Saved: {save_path}")
            print(f"Segments saved to {out_dir}")
            return

        # Otherwise, process notes files in the selected directory (batch mode for multiple trials)
        # If the selected raw_folder itself contains an RHD file and a notes file, process it as one trial
        main_notes_path = os.path.join(raw_folder, "notes.txt")
        processed_any = False
        if os.path.isfile(main_notes_path):
            # Find an RHD file in this folder (assuming one RHD per folder)
            rhd_files = [f for f in os.listdir(raw_folder) if f.endswith(".rhd")]
            if rhd_files:
                rhd_path = os.path.join(raw_folder, rhd_files[0])
                notes_df = load_labeled_file(path=main_notes_path)
                trial_name = os.path.basename(raw_folder)
                result = load_rhd_file(rhd_path)
                emg_data = result["amplifier_data"]
                fs = result["frequency_parameters"]["amplifier_sample_rate"]
                # Segment this single trial folder
                for i in range(len(notes_df) - 1):
                    label = notes_df.loc[i, "Label"]
                    if str(label).strip() == "" or str(label).lower() == "nan":
                        continue
                    start_idx = notes_df.loc[i, "Sample"]
                    end_idx = notes_df.loc[i + 1, "Sample"]
                    segment = emg_data[:, start_idx:end_idx]
                    label_safe = str(label).strip().replace(" ", "_").lower()
                    save_path = os.path.join(out_dir, f"{trial_name}_{label_safe}_{i}.npz")
                    np.savez_compressed(save_path, emg=segment, label=label, fs=fs)
                    print(f"Saved: {save_path}")
                processed_any = True

        # Process all subfolders in raw_folder (each subfolder is a trial folder)
        for folder_name in os.listdir(raw_folder):
            folder_path = os.path.join(raw_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue
            rhd_file = os.path.join(folder_path, f"{folder_name}.rhd")
            notes_file = os.path.join(folder_path, "notes.txt")
            if os.path.exists(rhd_file) and os.path.exists(notes_file):
                # Load data and notes for this trial
                print("file found containing labeled time indices. Processing:", folder_name)
                result = load_rhd_file(rhd_file)
                emg_data = result["amplifier_data"]
                fs = result["frequency_parameters"]["amplifier_sample_rate"]
                notes_df = load_labeled_file(path=notes_file)
                # Segment the trial data using note indices
                for i in range(len(notes_df) - 1):
                    label = notes_df.loc[i, "Label"]
                    if str(label).strip() == "" or str(label).lower() == "nan":
                        continue
                    start_idx = notes_df.loc[i, "Sample"]
                    end_idx = notes_df.loc[i + 1, "Sample"]
                    segment = emg_data[:, start_idx:end_idx]
                    label_safe = str(label).strip().replace(" ", "_").lower()
                    save_path = os.path.join(out_dir, f"{folder_name}_{label_safe}_{i}.npz")
                    np.savez_compressed(save_path, emg=segment, label=label, fs=fs)
                    print(f"Saved: {save_path}")
                processed_any = True

        if processed_any:
            print("Segmentation Complete", f"Segments saved to {out_dir}")
        else:
            print("No Segmentation Performed", "No .rhd and notes.txt pairs were found to segment.")

    def enable_indexing(self):
        self.indexing_enabled = True

    def scroll_plot(self, *args):
        if self.emg_data is None:
            return
        start_idx = int(float(args[1]) * len(self.time_vector))
        end_idx = start_idx + int(self.sampling_rate * 1)  # ~1s window
        self.ax.set_xlim(self.time_vector[start_idx], self.time_vector[min(end_idx, len(self.time_vector) - 1)])
        self.canvas.draw()

    def on_click(self, event):
        if not self.indexing_enabled or event.inaxes != self.ax:
            return
        sample_index = int(event.xdata * self.sampling_rate)
        label = self.label_entry.get()
        self.table.insert("", "end", values=(sample_index, label))
        self.ax.axvline(x=event.xdata, color='blue', linestyle='--')
        self.canvas.draw()
        self.indexing_enabled = False

    def save_table(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if not path:
            return
        with open(path, "w") as f:
            f.write("Sample Index,Label\n")
            for row in self.table.get_children():
                sample_index, label = self.table.item(row)["values"]
                f.write(f"{sample_index},{label}\n")
        print(f"Saved to {path}")

    def parse_channel_range(self, text):
        try:
            indices = []
            parts = text.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))
            return sorted(set(indices))
        except Exception as e:
            print(f"Error parsing channel range: {e}")
            return list(range(16))  # fallback

    def load_table(self):

        from intan.io import load_labeled_file
        df_sorted = load_labeled_file()

        # Update the table with the loaded data
        for row in self.table.get_children():
            self.table.delete(row)
        for _, row in df_sorted.iterrows():
            self.table.insert("", "end", values=(row["Sample"], row["Label"]))
        self.plot_channel()

    def delete_selected(self):
        for item in self.table.selection():
            self.table.delete(item)

    def update_channel(self, event=None):
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        self.current_channel = self.channel_selector.current()
        self.plot_channel()

    def plot_channel(self, event=None):

        self.ax.clear()
        if self.emg_data_raw is None:
            self.canvas.draw()
            return

        if self.domain_mode.get() == "Features":
            if self.segment_data is None:
                print("No segment loaded for 'Features' mode.")
                return

            ch_data = self.segment_data[self.current_channel]
            fs = self.segment_fs or 2000
            time = np.arange(len(ch_data)) / fs

            ch_data = self.apply_training_filters(ch_data)

            self.ax.plot(time, ch_data, label="Segment Ch {}".format(self.current_channel))

            # Overlay selected features (as text)
            features = self.extract_td_features(self.segment_data)
            text_lines = []
            if self.training_features["mean_absolute_value"].get():
                text_lines.append(f"MAV: {features[0]:.2f}")
            if self.training_features["zero_crossings"].get():
                zc_idx = 1 if "mean_absolute_value" in self.training_features and self.training_features[
                    "mean_absolute_value"].get() else 0
                text_lines.append(f"ZC: {features[zc_idx]:.0f}")

            self.ax.set_title("Features: " + ", ".join(text_lines))
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude (µV)")
            self.ax.grid(True)
            self.canvas.draw()
            return

        signal = self.emg_data_raw[self.current_channel]
        #signal = self.apply_visualization_filters(signal)
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print("Warning: filtered signal contains NaN or Inf values. They will be replaced with 0 for display.")
            signal = np.nan_to_num(signal)

        if self.domain_mode.get() == "Time":
            self.ax.plot(self.time_vector, signal, label=f"Ch {self.current_channel}")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude (µV)")
            self.ax.set_title("EMG Time Domain")

            # === Draw vertical lines for labeled trials ===
            if self.show_lines.get():
                for row in self.table.get_children():
                    sample_index, label = self.table.item(row)["values"]
                    t = sample_index / self.sampling_rate
                    self.ax.axvline(x=t, color="red", linestyle="--", linewidth=1)
                    self.ax.text(t + 0.05, self.ax.get_ylim()[1], label,
                                 rotation=90, verticalalignment="top", fontsize=8, color="red")

            if self.xlim:
                self.ax.set_xlim(self.xlim)
            if self.ylim:
                self.ax.set_ylim(self.ylim)

        elif self.domain_mode.get() == "PSD":
            N = len(signal)
            fs = self.sampling_rate
            frequencies = np.fft.rfftfreq(N, d=1 / fs)
            fft_magnitude = np.abs(np.fft.rfft(signal)) / N  # normalized magnitude
            self.ax.plot(frequencies, fft_magnitude)
            self.ax.set_ylabel("Magnitude")
            self.ax.set_title("EMG Power Spectral Density")
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_xlim(float(self.psd_min_freq.get()), float(self.psd_max_freq.get()))

        elif self.domain_mode.get() == "Spectrogram":
            try:
                nfft = int(self.nfft_entry.get())
                noverlap = int(self.noverlap_entry.get())
                cmap = self.cmap_entry.get()
                fs = self.sampling_rate

                # parse the signal with the time limits
                time_min = float(self.spec_time_range_min.get())
                time_max = float(self.spec_time_range_max.get())
                time_min_idx = int(time_min * fs)
                time_max_idx = int(time_max * fs)
                signal = signal[time_min_idx:time_max_idx]

                f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nfft, noverlap=noverlap)
                self.ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap)
                self.ax.set_ylabel("Frequency (Hz)")
                self.ax.set_xlabel("Time (s)")
                self.ax.set_title("EMG Spectrogram")
                self.ax.set_ylim(float(self.spec_freq_min.get()), float(self.spec_freq_max.get()))

            except Exception as e:
                self.ax.text(0.5, 0.5, f"Spectrogram error: {str(e)}", ha="center")

        elif self.domain_mode.get() == "Waterfall":
            ch_range_str = self.channel_range_entry.get()
            channel_indices = self.parse_channel_range(ch_range_str)
            self.waterfall_gui_plot(channel_indices)

        else:
            raise ValueError("Invalid domain mode selected.")

        self.canvas.draw()

    def extract_features_from_directories(self):
        """
        Iterate over each selected directory of segmented trials, apply filters,
        extract features, and aggregate the results. Saves individual feature files and builds a dataset.
        """
        all_features = []  # will collect feature vectors for dataset
        all_labels = []  # will collect corresponding labels

        # Loop through each directory added in the Training tab's directory list
        for item in self.dir_table.get_children():
            dir_path = self.dir_table.item(item)["values"][0]
            # Process each segmented .npz file in the directory
            for fname in os.listdir(dir_path):
                if not fname.endswith(".npz"):
                    continue
                file_path = os.path.join(dir_path, fname)
                data = np.load(file_path, allow_pickle=True)
                emg = data["emg"]  # segmented EMG data (channels x samples)
                label = data.get("label", "unknown").item()  # segment label

                # Use sampling rate from file if available, otherwise fallback to a default
                fs = data.get("fs", None)
                if fs is None:
                    fs = self.sampling_rate or 2000  # default to 2000 Hz if not specified

                # --- Apply selected preprocessing filters ---
                if self.filters["bandpass"].get():
                    # Bandpass filter (20-450 Hz) to remove motion artifacts and noise
                    #b, a = butter(4, [20, 450], btype="band", fs=fs)
                    #emg = filtfilt(b, a, emg, axis=1)
                    emg = bandpass_filter(emg, lowcut=20, highcut=450, fs=fs, order=4)
                if self.filters["notch"].get():
                    # Notch filter at 60 Hz to remove powerline interference
                    #b, a = iirnotch(60, Q=30, fs=fs)
                    #emg = filtfilt(b, a, emg, axis=1)
                    emg = notch_filter(emg, freq=60, fs=fs, Q=30)
                if self.filters["rectify"].get():
                    # Rectification (absolute value of the signal)
                    emg = np.abs(emg)
                if self.filters["smooth_rms"].get():
                    # Smoothing via moving window (200 ms RMS)
                    window_len = int(0.2 * fs)  # 200 ms window length in samples
                    if window_len > 1:
                        kernel = np.ones(window_len) / window_len
                        # Apply convolution along each channel (moving average = rough RMS)
                        emg = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=emg)

                # --- Extract selected features from the entire segment ---
                feature_vector = self.extract_td_features(emg)
                all_features.append(feature_vector)
                all_labels.append(label)

                # Save feature vector to a file (optional, for inspection or later use)
                feat_filename = os.path.join(dir_path, f"{os.path.splitext(fname)[0]}_feat.npy")
                np.save(feat_filename, feature_vector)
                # Note: We save as .npy (features only). The label is tracked separately in all_labels.

        # After processing all directories, build a combined dataset (features + labels)
        if not all_features:
            print("Warning: No features were extracted (check directory contents).")
            return

        features_array = np.vstack(all_features)  # shape: (num_samples, num_features)
        labels_array = np.array(all_labels, dtype=str)  # shape: (num_samples,)

        # Save the combined dataset to a CSV for easy viewing (features and label column)
        df = pd.DataFrame(features_array)
        df["Label"] = labels_array
        save_path = filedialog.asksaveasfilename(title="Save Training Dataset", defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Training dataset saved to {save_path}")
        else:
            # If user cancels save, still store the dataset in memory (e.g., as attributes)
            self.training_features = features_array
            self.training_labels = labels_array
            print("Training dataset has been created in memory.")

    def extract_td_features(self, segment):
        """ Computes a variety of unique features from segmented EMG data. Input shape expects (channels x samples). """
        features = []
        fs = self.segment_fs or self.sampling_rate or 2000

        for ch_data in segment:
            ch_features = []

            # Mean Absolute Value (MAV)
            if self.training_features.get("mean_absolute_value", tk.BooleanVar()).get():
                ch_features.append(np.mean(np.abs(ch_data)))

            # Zero Crossings (ZC)
            if self.training_features.get("zero_crossings", tk.BooleanVar()).get():
                try:
                    thresh = float(self.feature_controls["zero_crossings"]["threshold"].get())
                except (KeyError, ValueError):
                    thresh = 0.01
                zc = np.sum((np.diff(np.sign(ch_data)) != 0) & (np.abs(np.diff(ch_data)) > thresh))
                ch_features.append(zc)

            # Slope Sign Changes (SSC)
            if self.training_features.get("slope_sign_changes", tk.BooleanVar()).get():
                try:
                    delta_thresh = float(self.feature_controls["slope_sign_changes"]["delta_threshold"].get())
                except (KeyError, ValueError):
                    delta_thresh = 0.01
                diff = np.diff(ch_data)
                ssc = np.sum((np.diff(np.sign(diff)) != 0) & (np.abs(np.diff(diff)) > delta_thresh))
                ch_features.append(ssc)

            # Waveform Length (WL) - supports sub-windowed WL if segment is longer than window
            if self.training_features.get("waveform_length", tk.BooleanVar()).get():
                try:
                    window_ms = float(self.feature_controls["waveform_length"]["window_ms"].get())
                except (KeyError, ValueError):
                    window_ms = 200
                window_len = int((window_ms / 1000.0) * fs)

                if len(ch_data) < window_len:
                    wl = np.sum(np.abs(np.diff(ch_data)))
                else:
                    wl_vals = []
                    for start in range(0, len(ch_data) - window_len + 1, window_len):
                        sub_window = ch_data[start:start + window_len]
                        wl_vals.append(np.sum(np.abs(np.diff(sub_window))))
                    wl = np.mean(wl_vals)
                ch_features.append(wl)

            # RMS (used in Jehan Yang paper)
            if self.training_features.get("rms", tk.BooleanVar()).get():
                rms = np.sqrt(np.mean(ch_data ** 2))
                ch_features.append(rms)

            features.extend(ch_features)

        return np.array(features)

    def apply_training_filters(self, emg):
        """Applies selected filters to the EMG data. Handles both 1D (single channel) and 2D arrays."""
        fs = self.sampling_rate or 2000
        x = np.atleast_2d(emg)
        if x.shape[0] > x.shape[1]:
            x = x.T  # (channels, samples)

        if self.training_features.get("car", tk.BooleanVar(value=False)).get() and x.shape[0] > 1:
            x = common_average_reference(x)
        if self.training_features["bandpass"].get():
            x = bandpass_filter(x, lowcut=float(self.low_cut.get()), highcut=float(self.high_cut.get()),
                                fs=fs, order=int(self.filter_order.get()))
        if self.training_features["notch"].get():
            x = notch_filter(x, fs=fs, f0=60, Q=30, axis=1)
        if self.training_features["rectify"].get():
            x = rectify(x)
        if self.training_features["envelop_smooth"].get():
            try:
                envelop_f = float(self.smooth_f_entry.get())
            except ValueError:
                envelop_f = 5.0
            win_len = int(fs // envelop_f)
            if win_len > 1:
                x = window_rms(x, window_size=win_len)
        return np.squeeze(x)

    def add_scalebars(self, ax, scale_time=5, scale_voltage=10):
        ax.plot([0, scale_time], [-1000, -1000], color='gray', lw=3)
        ax.text(scale_time / 2, -1500, '5 sec', va='center', ha='center', fontsize=10, color='gray')
        ax.plot([0, 0], [-1000, -1000 + scale_voltage * 10], color='gray', lw=3)
        ax.text(-0.5, -500, '10 mV', va='center', ha='center', rotation='vertical', fontsize=10, color='gray')

    def insert_channel_labels(self, ax, time_vector, num_channels, num_labels=2, font_size=8):
        x_pos = time_vector[-1] + 1
        y_offsets = np.linspace(200, 25500, num_channels)
        ch_indices = np.linspace(0, num_channels - 1, num_labels, dtype=int)
        for ch in ch_indices:
            ax.text(x_pos, y_offsets[ch], f"Channel {ch}", fontsize=font_size, va='center',
                    ha='left', color='black', fontweight='bold')

    def insert_vertical_labels(self, ax):
        ax.text(-1, 5000, 'Extensor', fontsize=12, va='center', ha='center', color='black', rotation='vertical')
        ax.text(-1, 22000, 'Flexor', fontsize=12, va='center', ha='center', color='black', rotation='vertical')

    def build_training_dataset(self):
        # === Ask user for save location ===
        save_path = filedialog.asksaveasfilename(
            title="Save Training Dataset As",
            defaultextension=".npz",
            filetypes=[("NumPy Compressed", "*.npz")],
            initialfile="training_data.npz"
        )
        if not save_path:
            print("Dataset save cancelled.")
            return
        # Automatically derive PCA save path from dataset path
        pca_save_path = save_path.replace(".npz", "_pca.pkl")

        X = []  # Will hold the data features
        y = []  # Labels for the data
        fs = self.sampling_rate or 2000
        label_encoder = LabelEncoder()

        # === Loop over training_dir_table entries ===
        training_files = self.training_dir_table.get_children()
        for item in tqdm(training_files, desc="Extracting Features", position=0):
            file_path = self.training_dir_table.item(item)["values"][0]
            if not file_path.endswith(".npz") or not os.path.exists(file_path):
                continue

            # === Load the EMG data ===
            try:
                data = np.load(file_path)
                emg = data["emg"]
                label = data["label"].item() if "label" in data else "unknown"
            except Exception as e:
                print(f"Skipped {file_path}: could not load ({e})")
                continue

            # === Skip segments that are too short for filtering ===
            if emg.shape[1] < 27:
                print(f"Skipped {file_path}: segment too short ({emg.shape[1]} samples)")
                continue

            #print(f"Processing {file_path} with label: {label}")
            # === Apply training filters ======
            emg = self.apply_training_filters(emg)

            #print(f"\nEMG data shape: {emg.shape}")

            # === Extract features from the segment ===
            if self.training_features["use_sliding_window"].get():
                # Sliding window feature extraction
                window_size = int(self.window_size_entry.get())
                step_size = int(self.step_size_entry.get())

                if window_size <= 0 or step_size <= 0:
                    print("Error: Window and step size must be positive.")
                    return

                num_windows = (emg.shape[1] - window_size) // step_size + 1

                #print(f"Sliding window selected, number of window segments: {num_windows}")
                for i in tqdm(range(num_windows),
                              desc=f"{os.path.basename(file_path)}",
                              position=1, leave=False):
                    start = i * step_size
                    end = start + window_size
                    if end > emg.shape[1]:
                        break
                    segment = emg[:, start:end]
                    if segment.shape[1] != window_size:
                        continue

                    # Extract features from the segment
                    X.append(self.extract_td_features(segment))
                    y.append(label)

            else:
                # Single-segment feature extraction (entire EMG segment)
                X.append(self.extract_td_features(emg))
                y.append(label)

            #print(f"Current data size: {len(X)} segments")


        if not X:
            print("Error: No valid segments were processed.")
            return

        X = np.array(X)
        print("Shape of feature vectors:", X.shape)
        y_encoded = label_encoder.fit_transform(y)

        # === Normalize features ===
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero for constant features
        X = (X - mean) / std
        norm_save_path = save_path.replace(".npz", "_norm.npz")
        np.savez(norm_save_path, mean=mean, std=std)

        if X.shape[1] == 0:
            print("Error: Feature vectors are empty. Did you select any features?")
            return

        if not any(var.get() for var in self.training_features.values()):
            print("Warning: No features selected in GUI.")

        # === Apply PCA dimensionality reduction ===
        n_components = int(self.pca_components_entry.get()) if self.pca_components_entry.get().isdigit() else 50
        pca = PCA(n_components=n_components)

        pca.fit(X)
        joblib.dump(pca, pca_save_path)
        print(f"PCA model saved to: {pca_save_path}")
        X = pca.transform(X)  # now transform with saved-fit PCA

        print(f"New shape of feature vectors after PCA: {X.shape}")

        np.savez_compressed(save_path, features=X, labels=y_encoded, label_names=label_encoder.classes_)
        print(f"Training dataset saved to: {save_path}")

    def waterfall_gui_plot(self, channel_indices):
        signal = self.emg_data
        time_vector = self.time_vector
        ax = self.ax
        ax.clear()

        offset = 0
        offset_increment = 200
        cmap = plt.get_cmap("rainbow")
        num_channels = len(channel_indices)
        downsampling_factor = 1

        signal = signal[:, ::downsampling_factor]
        time_vector = time_vector[::downsampling_factor]

        for i, channel_idx in enumerate(channel_indices):
            if channel_idx < signal.shape[0]:
                channel_data = self.apply_training_filters(signal[channel_idx])
                color = cmap(i / num_channels)
                ax.plot(time_vector, channel_data + offset, color=color, linewidth=0.2)
                offset += offset_increment

        self.add_scalebars(ax)
        self.insert_vertical_labels(ax)
        self.insert_channel_labels(ax, time_vector, num_channels)

        ax.set_title("Waterfall Plot", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def on_closing(self):
        self.root.quit()


# Prefer the Qt implementation when the GUI extra is installed while keeping
# the Tk implementation available to callers that rely on it.
EMGViewer = EMGViewerQt if _HAS_PYQT else EMGViewerTk
