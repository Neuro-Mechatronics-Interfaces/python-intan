
import math
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from intan.processing import RealtimeFilter


# --- additions/changes inside your existing module ---

class FilterControls(QtWidgets.QWidget):
    """Dockable panel to control RealtimeFilter + view settings."""
    applyRequested = QtCore.pyqtSignal(dict)     # filter kwargs
    resetStateRequested = QtCore.pyqtSignal()    # filter state reset
    bypassToggled = QtCore.pyqtSignal(bool)      # True => show raw

    # NEW signals for view controls
    viewApplyRequested = QtCore.pyqtSignal(float, float)  # (ymin, ymax)
    autoYChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        ff = QtWidgets.QFormLayout(self)
        ff.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        # ---- Bypass
        self.chkBypass = QtWidgets.QCheckBox("Bypass (show raw)")
        ff.addRow(self.chkBypass)
        self.chkBypass.toggled.connect(self.bypassToggled.emit)

        # ---- Band-pass
        self.chkBP = QtWidgets.QCheckBox("Enable band-pass"); self.chkBP.setChecked(True)
        self.spBPlo = QtWidgets.QDoubleSpinBox(); self.spBPlo.setRange(0.1, 10_000); self.spBPlo.setValue(20.0)
        self.spBPhi = QtWidgets.QDoubleSpinBox(); self.spBPhi.setRange(0.1, 10_000); self.spBPhi.setValue(498.0)
        self.spBPord = QtWidgets.QSpinBox();      self.spBPord.setRange(1, 10); self.spBPord.setValue(4)
        ff.addRow(self.chkBP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Low (Hz):"));  row.addWidget(self.spBPlo)
        row.addWidget(QtWidgets.QLabel("High (Hz):")); row.addWidget(self.spBPhi)
        row.addWidget(QtWidgets.QLabel("Order:"));     row.addWidget(self.spBPord)
        ff.addRow(row)

        # ---- Notch
        self.chkNotch = QtWidgets.QCheckBox("Enable notch"); self.chkNotch.setChecked(True)
        self.leNotch = QtWidgets.QLineEdit("60, 120, 180")
        self.spQ = QtWidgets.QDoubleSpinBox(); self.spQ.setRange(1.0, 200.0); self.spQ.setValue(30.0)
        ff.addRow(self.chkNotch)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Freqs (Hz):")); row.addWidget(self.leNotch)
        row.addWidget(QtWidgets.QLabel("Q:"));          row.addWidget(self.spQ)
        ff.addRow(row)

        # ---- Low-pass
        self.chkLP = QtWidgets.QCheckBox("Enable low-pass (post)"); self.chkLP.setChecked(False)
        self.spLPcut = QtWidgets.QDoubleSpinBox(); self.spLPcut.setRange(0.1, 10_000); self.spLPcut.setValue(300.0)
        self.spLPord = QtWidgets.QSpinBox();      self.spLPord.setRange(1, 10); self.spLPord.setValue(4)
        ff.addRow(self.chkLP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cutoff (Hz):")); row.addWidget(self.spLPcut)
        row.addWidget(QtWidgets.QLabel("Order:"));       row.addWidget(self.spLPord)
        ff.addRow(row)

        # ---- Filter buttons
        btnRow = QtWidgets.QHBoxLayout()
        self.btnApply = QtWidgets.QPushButton("Apply filter")
        self.btnReset = QtWidgets.QPushButton("Reset state")
        btnRow.addWidget(self.btnApply); btnRow.addStretch(1); btnRow.addWidget(self.btnReset)
        ff.addRow(btnRow)

        self.btnApply.clicked.connect(self._emit_apply)
        self.btnReset.clicked.connect(self.resetStateRequested.emit)

        # ===== NEW: View controls (Y range) =====
        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine)
        ff.addRow(line)

        self.chkAutoY = QtWidgets.QCheckBox("Auto Y (robust)"); self.chkAutoY.setChecked(False)
        ff.addRow(self.chkAutoY)
        self.chkAutoY.toggled.connect(self.autoYChanged.emit)

        row = QtWidgets.QHBoxLayout()
        self.spYmin = QtWidgets.QDoubleSpinBox(); self.spYmin.setRange(-1e6, 1e6); self.spYmin.setDecimals(1); self.spYmin.setValue(-500.0)
        self.spYmax = QtWidgets.QDoubleSpinBox(); self.spYmax.setRange(-1e6, 1e6); self.spYmax.setDecimals(1); self.spYmax.setValue(500.0)
        row.addWidget(QtWidgets.QLabel("Y min (µV):")); row.addWidget(self.spYmin)
        row.addWidget(QtWidgets.QLabel("Y max (µV):")); row.addWidget(self.spYmax)
        ff.addRow(row)

        self.btnViewApply = QtWidgets.QPushButton("Apply view")
        ff.addRow(self.btnViewApply)
        self.btnViewApply.clicked.connect(lambda: self.viewApplyRequested.emit(self.spYmin.value(), self.spYmax.value()))

    def _emit_apply(self):
        txt = self.leNotch.text().replace(",", " ")
        freqs = [float(tok) for tok in txt.split() if tok.strip()]
        kwargs = dict(
            enable_bandpass=self.chkBP.isChecked(),
            bp_low=float(self.spBPlo.value()),
            bp_high=float(self.spBPhi.value()),
            bp_order=int(self.spBPord.value()),
            enable_notch=self.chkNotch.isChecked(),
            notch_freqs=tuple(freqs),
            notch_q=float(self.spQ.value()),
            enable_lowpass=self.chkLP.isChecked(),
            lp_cut=float(self.spLPcut.value()),
            lp_order=int(self.spLPord.value()),
        )
        self.applyRequested.emit(kwargs)


class StackedPlot(QtWidgets.QMainWindow):
    def __init__(
        self,
        client,
        window_secs=5.0,
        ui_hz=20,
        auto_ylim=False,
        y_limits=(-500.0, 500.0),
        robust_pct=(1, 99),
        min_span=1e-6,
        smooth_alpha=0.25,
        symmetric=False,
        max_points=2000,
        enable_filter_ui=True,
    ):
        super().__init__()
        self.client = client
        self.window_secs = float(window_secs)
        self.auto_ylim = bool(auto_ylim)
        self.fixed_ylim = tuple(y_limits) if y_limits is not None else None
        self.robust_pct = robust_pct
        self.min_span = float(min_span)
        self.smooth_alpha = float(smooth_alpha)
        self.symmetric = bool(symmetric)
        self.max_points = int(max_points)
        self.n_channels = self.client.n_channels

        self.setWindowTitle(
            f"Stacked — {self.client.name}  type={self.client.type}  "
            f"nom fs={self.client.fs:.1f}Hz  chans={self.n_channels}"
        )
        self.resize(1200, 820)

        # ---- central scrollable area with vertical layout ----
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(6, 6, 6, 6); vbox.setSpacing(4)

        self.scroll = QtWidgets.QScrollArea(); self.scroll.setWidgetResizable(True)
        self.inner = QtWidgets.QWidget()
        self.inner_layout = QtWidgets.QVBoxLayout(self.inner)
        self.inner_layout.setContentsMargins(0, 0, 0, 0); self.inner_layout.setSpacing(2)

        # Construct plots
        self.plots, self.curves = [], []
        self._ylims = [None] * self.n_channels
        for i in range(self.n_channels):
            pw = pg.PlotWidget()
            pw.showGrid(x=True, y=True)
            pw.setXRange(-self.window_secs, 0)
            if i < self.n_channels - 1:
                pw.hideAxis("bottom")
            label = (
                self.client.channel_labels[i]
                if hasattr(self.client, "channel_labels") and i < len(self.client.channel_labels)
                else f"{i}"
            )
            pw.setLabel("left", f"Ch {label}")

            #pw.setLabel("left", f"Ch {self.client.channel_labels[i]}")
            curve = pw.plot(pen=pg.mkPen(pg.intColor(i, hues=self.n_channels), width=1))
            if self.fixed_ylim is not None:
                pw.setYRange(self.fixed_ylim[0], self.fixed_ylim[1], padding=0)
            self.inner_layout.addWidget(pw)
            self.plots.append(pw); self.curves.append(curve)

        for i in range(1, self.n_channels):
            self.plots[i].setXLink(self.plots[0])

        self.scroll.setWidget(self.inner)
        vbox.addWidget(self.scroll)
        self.setCentralWidget(central)

        # ---- Realtime filter
        self._bypass = False
        self.filter = RealtimeFilter(
            fs=self.client.fs,
            n_channels=self.n_channels,
        )

        # self._ft = np.zeros(self.client.T, dtype=np.float64)  # filter state for each channel
        # self._fy = np.zeros((self.n_channels, self.client.T), dtype=np.float32)  # filtered output
        # self._fwidx = 0
        # self._fcount = 0

        # ---- Optional filter UI
        if enable_filter_ui:
            self._add_filter_dock()

        # timers
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / ui_hz))

        self.fs_timer = QtCore.QTimer()
        self.fs_timer.timeout.connect(self._update_title_fs)
        self.fs_timer.start(1000)

    # ---------- UI bits ----------
    def _add_filter_dock(self):
        dock = QtWidgets.QDockWidget("Filter", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.controls = FilterControls()
        dock.setWidget(self.controls)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # Filtering signals
        self.controls.applyRequested.connect(self._on_apply_filter)
        self.controls.resetStateRequested.connect(self.filter.reset)
        self.controls.bypassToggled.connect(self._on_bypass)

        # Viewing signals
        self.controls.viewApplyRequested.connect(self._on_view_apply)
        self.controls.autoYChanged.connect(self._on_auto_y)

    def _on_view_apply(self, ymin: float, ymax: float):
        if ymin >= ymax:
            QtWidgets.QMessageBox.warning(self, "View error", "Y min must be < Y max.")
            return
        self.fixed_ylim = (ymin, ymax)
        self.auto_ylim = False
        for pw in self.plots:
            pw.setYRange(ymin, ymax, padding=0)

    def _on_auto_y(self, enabled: bool):
        self.auto_ylim = bool(enabled)
        # when turning auto on, clear smoothed limits so it re-initializes
        if self.auto_ylim:
            self._ylims = [None] * self.n_channels
        else:
            # when turning it off, snap to current fixed range if set
            if self.fixed_ylim is not None:
                for pw in self.plots:
                    pw.setYRange(self.fixed_ylim[0], self.fixed_ylim[1], padding=0)

    def _on_apply_filter(self, kw):
        try:
            self.filter.reconfigure(**kw)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Filter config error", str(e))

    def _on_bypass(self, state: bool):
        self._bypass = bool(state)

    # ---------- window plumbing ----------
    def closeEvent(self, ev):
        self.client.stop()
        super().closeEvent(ev)

    def _update_title_fs(self):
        fs_hat = getattr(self.client, "fs_estimate", lambda: float("nan"))()
        title = (
            f"LSL Stacked — {getattr(self.client, 'name', 'Stream')}  "
            f"type={getattr(self.client, 'type', '?')}  "
            f"nom fs={getattr(self.client, 'fs', float('nan')):.1f}Hz"
        )
        if np.isfinite(fs_hat):
            title += f"  est fs={fs_hat:.2f}Hz"
        title += f"  chans={self.n_channels}"
        self.setWindowTitle(title)

    def _robust_limits(self, y):
        lo, hi = np.nanpercentile(y, self.robust_pct)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if self.symmetric:
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        span = max(self.min_span, hi - lo)
        pad = 0.10 * span
        return lo - pad, hi + pad

    def _smooth_set_ylim(self, idx, lo, hi):
        if self._ylims[idx] is None:
            self._ylims[idx] = [lo, hi]
        else:
            a = self.smooth_alpha
            self._ylims[idx][0] = (1 - a) * self._ylims[idx][0] + a * lo
            self._ylims[idx][1] = (1 - a) * self._ylims[idx][1] + a * hi
        self.plots[idx].setYRange(self._ylims[idx][0], self._ylims[idx][1], padding=0)

    def _maybe_decimate(self, t, y):
        n = t.size
        if n <= self.max_points:
            return t, y
        step = int(math.ceil(n / self.max_points))
        return t[::step], y[..., ::step]

    # ---------- main update ----------
    def update_plots(self):
        t_rel, Y = self.client.latest(window_secs=self.window_secs)
        if t_rel is None or Y.size == 0:
            return

        if not self._bypass:
            Y = self.filter.process(Y)

        t_rel, Y = self._maybe_decimate(t_rel, Y)
        for i in range(self.n_channels):
            yi = Y[i]
            self.curves[i].setData(t_rel, yi)
            if self.auto_ylim and yi.size:
                lims = self._robust_limits(yi)
                if lims is not None:
                    self._smooth_set_ylim(i, *lims)

    # def update_plots(self):
    #     # get only new samples
    #     t_new, Y_new = self.client.drain_new()
    #     if t_new is not None and Y_new.shape[1] > 0:
    #         # filter just the new tail (or bypass)
    #         if not getattr(self, "_bypass", False):
    #             Y_new = self.filter.process(Y_new)
    #
    #         # append to filtered ring
    #         n = Y_new.shape[1]
    #         dst = self._fwidx % self.client.n_samples
    #         first = min(n, self.client.n_samples - dst)
    #         self._fy[:, dst:dst + first] = Y_new[:, :first]
    #         self._ft[dst:dst + first] = t_new[:first]
    #         rem = n - first
    #         if rem > 0:
    #             self._fy[:, :rem] = Y_new[:, first:]
    #             self._ft[:rem] = t_new[first:]
    #         self._fwidx = (self._fwidx + n) % self.client.n_samples
    #         self._fcount = min(self._fcount + n, self.client.n_samples)
    #
    #     # nothing to draw yet
    #     if self._fcount == 0:
    #         return
    #
    #     # build last window from filtered ring (exactly like client.latest, but on filtered data)
    #     end = self._fwidx
    #     if self._fcount < self.client.n_samples:
    #         t = self._ft[:self._fcount].copy()
    #         Y = self._fy[:, :self._fcount].copy()
    #     else:
    #         t = np.hstack((self._ft[end:], self._ft[:end])).copy()
    #         Y = np.hstack((self._fy[:, end:], self._fy[:, :end])).copy()
    #
    #     t_last = t[-1]
    #     t_rel = t - t_last
    #     mask = t_rel >= -self.window_secs
    #     t_rel, Y = t_rel[mask], Y[:, mask]
    #
    #     # (optional) decimate for draw speed
    #     t_rel, Y = self._maybe_decimate(t_rel, Y)
    #
    #     for i in range(self.n_channels):
    #         yi = Y[i]
    #         self.curves[i].setData(t_rel, yi)
    #         if self.auto_ylim and yi.size:
    #             lims = self._robust_limits(yi)
    #             if lims is not None:
    #                 self._smooth_set_ylim(i, *lims)


    # def update_plots(self):
    #     t_rel, Y = self.client.latest()
    #     if t_rel is None:
    #         return
    #
    #     # filter (stateful) unless bypassed
    #     if not self._bypass:
    #         Y = self.filter.process(Y)
    #
    #     # decimate for draw speed
    #     t_rel, Y = self._maybe_decimate(t_rel, Y)
    #
    #     for i in range(self.n_channels):
    #         yi = Y[i]
    #         self.curves[i].setData(t_rel, yi)
    #         if self.auto_ylim and yi.size:
    #             lims = self._robust_limits(yi)
    #             if lims is not None:
    #                 self._smooth_set_ylim(i, *lims)
