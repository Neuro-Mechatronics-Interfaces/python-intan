#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, sys, socket
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
import pyqtgraph as pg
#from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_streams, StreamInlet, IRREGULAR_RATE
from intan.interface import LSLClient, IntanRHXDevice, LSLOptions


APP_TITLE = "Gesture Pipeline Control"
DEFAULT_PROFILE_NAME = "gesture_pipeline_profile.json"

# ---------------------------------------------------------------------------
# Style (mac-ish light + optional dark)
# ---------------------------------------------------------------------------
class Style:
    @staticmethod
    def apply(app: QtWidgets.QApplication, *, dark=False, accent="#0A84FF"):
        # Keep native on macOS; otherwise use Fusion
        if sys.platform != "darwin":
            app.setStyle("Fusion")

        pal = QtGui.QPalette()
        if not dark:
            pal.setColor(pal.Window,          QtGui.QColor("#F6F7F9"))
            pal.setColor(pal.Base,            QtGui.QColor("#FFFFFF"))
            pal.setColor(pal.AlternateBase,   QtGui.QColor("#F1F2F6"))
            pal.setColor(pal.Text,            QtGui.QColor("#1C1C1E"))
            pal.setColor(pal.Button,          QtGui.QColor("#FFFFFF"))
            pal.setColor(pal.ButtonText,      QtGui.QColor("#1C1C1E"))
            pal.setColor(pal.Highlight,       QtGui.QColor(accent))
            pal.setColor(pal.HighlightedText, QtGui.QColor("#FFFFFF"))
            pal.setColor(pal.PlaceholderText, QtGui.QColor("#9AA0A6"))
            btn_border, btn_bg, btn_fg = "#D0D5DD", "#FFFFFF", "#1C1C1E"
            btn_hover, btn_press = "#F7F8FA", "#F0F2F5"
            line_border = "#E5E7EB"
            label_color = "#1C1C1E"
        else:
            pal.setColor(pal.Window,          QtGui.QColor("#1E1F22"))
            pal.setColor(pal.Base,            QtGui.QColor("#121316"))
            pal.setColor(pal.AlternateBase,   QtGui.QColor("#1A1B1E"))
            pal.setColor(pal.Text,            QtGui.QColor("#E5E7EB"))
            pal.setColor(pal.Button,          QtGui.QColor("#2A2C30"))
            pal.setColor(pal.ButtonText,      QtGui.QColor("#E5E7EB"))
            pal.setColor(pal.Highlight,       QtGui.QColor(accent))
            pal.setColor(pal.HighlightedText, QtGui.QColor("#FFFFFF"))
            pal.setColor(pal.PlaceholderText, QtGui.QColor("#8B8F98"))
            btn_border, btn_bg, btn_fg = "#4A4D55", "#2A2C30", "#E5E7EB"
            btn_hover, btn_press = "#32343A", "#2A2C30"
            line_border = "#3A3D44"
            label_color = "#E5E7EB"

        app.setPalette(pal)

        font = app.font()
        font.setPointSizeF(max(10.0, font.pointSizeF() + 0.5))
        if sys.platform == "win32":
            font.setFamily("Segoe UI Variable")
        app.setFont(font)

        app.setStyleSheet(f"""
                    QTabWidget::pane {{ border: 0; }}
                    QTabBar::tab {{ padding: 6px 12px; margin-right: 2px; color:{label_color}; }}
                    QGroupBox {{ border: 0; margin-top: 10px; }}
                    QGroupBox::title {{ font-weight: 600; padding-bottom: 6px; color:{label_color}; }}
                    QLabel {{ color:{label_color}; }}
                    QLabel[hint="true"] {{ color: #6B7280; font-size: 12px; }}
                    QFrame#Card, QTextEdit#LogPane, QPlainTextEdit#LogPane {{
                        background: palette(Base); border: 1px solid {line_border}; border-radius: 10px;
                    }}
                    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
                        padding: 6px 8px; border: 1px solid {line_border}; border-radius: 8px;
                        color: {btn_fg};
                    }}
                    QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                        border: 1px solid {accent};
                    }}
                    QPushButton {{
                        padding: 7px 12px; border-radius: 10px;
                        border: 1px solid {btn_border};
                        background: {btn_bg};
                        color: {btn_fg};
                    }}
                    QPushButton:hover {{ background: {btn_hover}; }}
                    QPushButton:pressed {{ background: {btn_press}; }}
                    QPushButton[accent="true"] {{ background: {accent}; color: white; border: 1px solid {accent}; }}
                    QPushButton[accent="true"]:disabled {{ background: #9CCBFF; border-color: #9CCBFF; color: #F3F8FF; }}
                    QStatusBar {{ padding: 0 8px; }}
                """)

        # app.setStyleSheet(f"""
        #     /* ... keep your existing rules ... */
        #
        #     /* Label/title color certainty in dark mode */
        #     {'QLabel, QGroupBox::title { color: #E5E7EB; }' if dark else ''}
        #
        #     QFrame#Card, QTextEdit#LogPane, QPlainTextEdit#LogPane {{
        #         background: palette(Base); border: 1px solid {'#3A3D44' if dark else '#E5E7EB'}; border-radius: 10px;
        #     }}
        #     QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        #         padding: 6px 8px; border: 1px solid {'#3A3D44' if dark else '#E5E7EB'}; border-radius: 8px;
        #         color: {('#E5E7EB' if dark else '#1C1C1E')};
        #         background: {('#1A1B1E' if dark else '#FFFFFF')};
        #     }}
        #     QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        #         border: 1px solid {accent};
        #     }}
        #
        #     /* Bordered buttons, with hover and pressed states */
        #     QPushButton {{
        #         padding: 7px 12px; border-radius: 10px;
        #         border: 1px solid {'#4A4D55' if dark else '#D0D5DD'};
        #         background: {('#2A2C30' if dark else '#FFFFFF')};
        #         color: {('#E5E7EB' if dark else '#1C1C1E')};
        #     }}
        #     QPushButton:hover {{
        #         background: {('#32343A' if dark else '#F7F8FA')};
        #     }}
        #     QPushButton:pressed {{
        #         background: {('#2A2C30' if dark else '#F0F2F5')};
        #     }}
        #
        #     /* Accent (primary) buttons */
        #     QPushButton[accent="true"] {{
        #         background: {accent}; color: white;
        #         border: 1px solid {accent};
        #     }}
        #     QPushButton[accent="true"]:hover {{
        #         filter: brightness(1.05);
        #     }}
        #     QPushButton[accent="true"]:disabled {{
        #         background: #9CCBFF; border-color: #9CCBFF; color: #F3F8FF;
        #     }}
        #
        #     /* Tabs */
        #     QTabBar::tab {{
        #         padding: 6px 12px; margin-right: 2px; border: 1px solid transparent; border-bottom: 0;
        #         color: {('#E5E7EB' if dark else '#1C1C1E')};
        #     }}
        #     QTabBar::tab:selected {{
        #         background: {('#1E1F22' if dark else '#FFFFFF')};
        #         border-color: {'#3A3D44' if dark else '#E5E7EB'};
        #         border-top-left-radius: 10px; border-top-right-radius: 10px;
        #     }}
        # """)


# ---------------------------------------------------------------------------
# Utilities that mirror your originals
# ---------------------------------------------------------------------------
def which_python() -> str:
    return sys.executable or "python"

def default_examples_dir() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parent, here.parent.parent, here.parents[2] if len(here.parents) > 2 else here.parent):
        p = base / "examples" / "gesture_classifier"
        if p.exists():
            return p
    return here.parent

def default_scripts() -> dict:
    eg = default_examples_dir()
    return {
        "build": str(eg / "1a_build_training_dataset_rhd.py"),
        "train": str(eg / "2_train_model.py"),
        "realtime": str(eg / "3d_predict_from_device_realtime.py"),
    }

# ----------------------------------------------------------------------------
# SVG helpers
# ----------------------------------------------------------------------------

def _icon(svg: str, size: int = 20) -> QtGui.QIcon:
    """Make a QIcon from a tiny inline SVG string (monochrome)."""
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    renderer.render(p)
    p.end()
    return QtGui.QIcon(pm)

# 1-color SVGs (simple, neutral). Stroke inherits currentColor; we set fill.
SVG_COLOR_LIGHT = "#1C1C1E"   # used in light mode
SVG_COLOR_DARK  = "#E5E7EB"   # used in dark mode

def svg_code(color):  # Scripts
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <path d='M9.5 6.5 5 12l4.5 5.5' stroke='{color}' stroke-width='1.6'
              stroke-linecap='round' stroke-linejoin='round'/>
        <path d='M14.5 6.5 19 12l-4.5 5.5' stroke='{color}' stroke-width='1.6'
              stroke-linecap='round' stroke-linejoin='round'/>
    </svg>"""

def svg_folder(color):  # Dataset
    return f"""<svg width="24" height="24" viewBox="0 0 24 24" fill="none"
        xmlns='http://www.w3.org/2000/svg'>
        <path d="M3 6.5h5.2l1.6 2H21v8.5a2 2 0 0 1-2 2H5
                 a2 2 0 0 1-2-2V6.5Z" fill="{color}" fill-opacity="0.12"/>
        <path d="M3 7.5V6a2 2 0 0 1 2-2h3.2l1.6 2H21a2 2 0 0 1 2 2v9.5
                 a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V7.5Z"
              stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>"""

def svg_gear(color):    # Train
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <circle cx='12' cy='12' r='3.5' stroke='{color}' stroke-width='1.6'/>
        <path d='M12 2.8v2.4M12 18.8v2.4M21.2 12h-2.4M5.2 12H2.8
                 M17.5 6.5l-1.7 1.7M8.2 15.8 6.5 17.5M17.5 17.5l-1.7-1.7M8.2 8.2 6.5 6.5'
              stroke='{color}' stroke-width='1.6' stroke-linecap='round'/>
    </svg>"""

def svg_play(color):    # Predict
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <path d='M8 6l10 6-10 6V6Z' fill='{color}' fill-opacity='0.14' />
        <path d='M8 6l10 6-10 6V6Z' stroke='{color}' stroke-width='1.6' stroke-linejoin='round'/>
    </svg>"""

def svg_live(color):    # Live
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <circle cx='12' cy='12' r='3.2' fill='{color}'/>
        <circle cx='12' cy='12' r='7.5' stroke='{color}' stroke-width='1.6' opacity='0.55'/>
    </svg>"""

def svg_rms(color):     # Channel View
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <rect x='5' y='10' width='2.2' height='8' fill='{color}' opacity='0.85'/>
        <rect x='9' y='6'  width='2.2' height='12' fill='{color}' opacity='0.85'/>
        <rect x='13' y='12' width='2.2' height='6' fill='{color}' opacity='0.85'/>
        <rect x='17' y='8'  width='2.2' height='10' fill='{color}' opacity='0.85'/>
        <rect x='3' y='3' width='4' height='4' stroke='{color}' stroke-width='1.4' opacity='0.7'/>
    </svg>"""

# ---------------------------------------------------------------------------
# QProcess-based runner (native, responsive)
# ---------------------------------------------------------------------------
class ProcRunner(QtCore.QObject):
    started  = QtCore.pyqtSignal(str)
    line     = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(int)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None

    def run(self, cmd, cwd=None, env=None):
        if self.proc and self.proc.state() != QtCore.QProcess.NotRunning:
            self.failed.emit("A process is already running.")
            return
        self.started.emit("$ " + " ".join(cmd))
        self.proc = QtCore.QProcess(self)
        if cwd:
            self.proc.setWorkingDirectory(cwd)

        qenv = QtCore.QProcessEnvironment.systemEnvironment()
        qenv.insert("PYTHONIOENCODING", "utf-8")
        if env:
            for k, v in env.items():
                qenv.insert(k, v)
        self.proc.setProcessEnvironment(qenv)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_out)
        self.proc.finished.connect(self._on_finished)
        try:
            self.proc.start(cmd[0], cmd[1:])
        except Exception as e:
            self.failed.emit(f"Failed to start: {e}")

    def _on_out(self):
        if not self.proc:
            return
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for ln in text.splitlines():
            self.line.emit(ln)

    def _on_finished(self, code, _status):
        self.proc = None
        self.finished.emit(int(code))

    def stop(self):
        if self.proc and self.proc.state() != QtCore.QProcess.NotRunning:
            self.proc.terminate()
            if not self.proc.waitForFinished(1500):
                self.proc.kill()

# ──────────────────────────────────────────────────────────────────────────────
# Channel Viewer Tab
# -─────────────────────────────────────────────────────────────────────────────
def _cell(title: str, widget: QtWidgets.QWidget, maxw: int | None = None) -> QtWidgets.QWidget:
    wrap = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(wrap)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    lbl = QtWidgets.QLabel(title)
    lbl.setProperty("hint", True)  # uses your subtle label color
    v.addWidget(lbl)
    if maxw:
        widget.setMaximumWidth(maxw)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    v.addWidget(widget)
    return wrap

class ChannelViewWidget(QtWidgets.QWidget):
    """
    Live RMS viewer for EMG channels from an LSL numeric stream.

    - Connect/disconnect to LSL stream
    - Choose channel count (default 128)
    - Enable/disable channels (checkbox list with quick selects)
    - Two display modes: "RMS Bars" and "RMS Grid"
    - Emits selectedChannelsChanged(list[int]) so MainWindow can carry them to Dataset tab
    """
    selectedChannelsChanged = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ----------------- UI -----------------
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        card = QtWidgets.QFrame(objectName="Card")
        grid = QtWidgets.QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        r = 0
        # Row 0: Source + common fields
        self.cmbSource = QtWidgets.QComboBox()
        self.cmbSource.addItems(["LSL stream", "Intan device (direct)"])

        self.spnCh = QtWidgets.QSpinBox();
        self.spnCh.setRange(1, 1024);
        self.spnCh.setValue(128)
        self.dsbWin = QtWidgets.QDoubleSpinBox();
        self.dsbWin.setRange(0.05, 10.0);
        self.dsbWin.setValue(0.25);
        self.dsbWin.setSuffix(" s")
        self.cmbMode = QtWidgets.QComboBox();
        self.cmbMode.addItems(["RMS Bars", "RMS Grid"])

        # --- Threshold UI (same row as RMS window) ---
        self.dsbThresh = QtWidgets.QDoubleSpinBox()
        self.dsbThresh.setRange(0.0, 5000.0)
        self.dsbThresh.setDecimals(2)
        self.dsbThresh.setSingleStep(5.0)
        self.dsbThresh.setValue(30.0)
        self.dsbThresh.setSuffix(" µV")

        self.btnEnableCross = QtWidgets.QPushButton("Enable > threshold")
        self.btnDisableCross = QtWidgets.QPushButton("Disable > threshold")

        thr_row = QtWidgets.QHBoxLayout()
        thr_row.setContentsMargins(0, 0, 0, 0)
        thr_row.setSpacing(8)
        thr_row.addWidget(self.dsbThresh)
        thr_row.addWidget(self.btnEnableCross)
        thr_row.addWidget(self.btnDisableCross)
        thr_wrap = QtWidgets.QWidget();
        thr_wrap.setLayout(thr_row)

        grid.addWidget(_cell("Source", self.cmbSource, maxw=220), r, 0)
        grid.addWidget(_cell("Channels", self.spnCh, maxw=140), r, 1)
        grid.addWidget(_cell("RMS window", self.dsbWin, maxw=160), r, 2)
        grid.addWidget(_cell("Threshold", thr_wrap, maxw=360), 0, 3)
        r += 1

        # Row 1: LSL fields (shown for 'LSL stream')
        self.leType = QtWidgets.QLineEdit("EMG")
        self.leName = QtWidgets.QLineEdit("")
        lsl_row = QtWidgets.QHBoxLayout()
        lsl_row.setContentsMargins(0, 0, 0, 0);
        lsl_row.setSpacing(8)
        lsl_row.addWidget(QtWidgets.QLabel("Type:"));
        lsl_row.addWidget(self.leType)
        lsl_row.addWidget(QtWidgets.QLabel("Name:"));
        lsl_row.addWidget(self.leName)
        self._wLSL = QtWidgets.QWidget();
        self._wLSL.setLayout(lsl_row)
        grid.addWidget(_cell("LSL", self._wLSL), r, 0, 1, 2)

        # Connect/disconnect buttons (common)
        self.btnConnect = QtWidgets.QPushButton("Connect");
        self.btnConnect.setProperty("accent", True)
        self.btnDisconnect = QtWidgets.QPushButton("Disconnect");
        self.btnDisconnect.setEnabled(False)
        btns = QtWidgets.QHBoxLayout();
        btns.setContentsMargins(0, 0, 0, 0);
        btns.setSpacing(8)
        btns.addStretch(1);
        btns.addWidget(self.btnConnect);
        btns.addWidget(self.btnDisconnect)
        btnsW = QtWidgets.QWidget();
        btnsW.setLayout(btns)
        grid.addWidget(btnsW, r, 2)
        r += 1

        # Row 2: Device fields (shown for 'Intan device (direct)')
        self.leHost = QtWidgets.QLineEdit("127.0.0.1")
        self.leCmd = QtWidgets.QLineEdit("5000")
        self.leData = QtWidgets.QLineEdit("5001")
        dev_row = QtWidgets.QHBoxLayout()
        dev_row.setContentsMargins(0, 0, 0, 0);
        dev_row.setSpacing(8)
        dev_row.addWidget(QtWidgets.QLabel("Host:"));
        dev_row.addWidget(self.leHost)
        dev_row.addWidget(QtWidgets.QLabel("Cmd:"));
        self.leCmd.setMaximumWidth(90);
        dev_row.addWidget(self.leCmd)
        dev_row.addWidget(QtWidgets.QLabel("Data:"));
        self.leData.setMaximumWidth(90);
        dev_row.addWidget(self.leData)
        self._wDev = QtWidgets.QWidget();
        self._wDev.setLayout(dev_row)
        grid.addWidget(_cell("Device (Intan)", self._wDev), r, 0, 1, 3)
        r += 1

        # Channel selector (unchanged)
        self.listCh = QtWidgets.QListWidget()
        self.listCh.setMaximumHeight(140)
        self._rebuild_channel_list(self.spnCh.value())

        mini_btns = QtWidgets.QHBoxLayout()
        self.btnAll = QtWidgets.QPushButton("All")
        self.btnNone = QtWidgets.QPushButton("None")
        self.btnEven = QtWidgets.QPushButton("Even")
        self.btnOdd = QtWidgets.QPushButton("Odd")
        [mini_btns.addWidget(x) for x in (self.btnAll, self.btnNone, self.btnEven, self.btnOdd)]
        mini_btns.addStretch(1)

        grid.addWidget(QtWidgets.QLabel("Select channels"), r, 0, QtCore.Qt.AlignTop)
        grid.addWidget(self.listCh, r, 1, 1, 2)
        r += 1
        grid.addLayout(mini_btns, r, 1, 1, 2)
        r += 1

        self.btnCarry = QtWidgets.QPushButton("Use selection in Dataset")
        grid.addWidget(self.btnCarry, r, 0, 1, 3)
        root.addWidget(card)

        # Plot stack (unchanged, but make taller by default)
        self.stack = QtWidgets.QStackedLayout()
        self._barPlot = pg.PlotWidget();
        self._barPlot.showGrid(x=True, y=True)
        self._barBars = None
        self._imgPlot = pg.PlotWidget();
        self._imgItem = pg.ImageItem(autoDownsample=True)
        self._imgPlot.addItem(self._imgItem);
        self._imgPlot.invertY(True);
        self._imgPlot.showGrid(x=True, y=True)

        # Threshold line for bar plot
        # self._thLine = pg.InfiniteLine(pos=self.dsbThresh.value(), angle=0)
        # self._barPlot.addItem(self._thLine)
        # self.dsbThresh.valueChanged.connect(lambda v: self._thLine.setPos(float(v)))

        # scratch buffers for last window
        self._last_rms = None  # np.ndarray, shape (n_selected,)
        self._last_sel_idx = None  # np.ndarray of selected channel indices (global)
        self._barBarsLo = None
        self._barBarsHi = None

        self.stackW = QtWidgets.QWidget();
        self.stackW.setLayout(self.stack)
        self.stack.addWidget(self._barPlot);
        self.stack.addWidget(self._imgPlot)
        self.stackW.setMinimumHeight(360)  # <— taller default
        root.addWidget(self.stackW, 1)

        # ----------------- state/wires -----------------
        self.client = None  # LSLClient
        self._rhx = None  # IntanRHXDevice (direct)
        self.timer = QtCore.QTimer(self);
        self.timer.timeout.connect(self._tick)
        self.ui_hz = 20

        self.spnCh.valueChanged.connect(self._on_chcount_changed)
        self.cmbMode.currentIndexChanged.connect(self._on_mode_changed)
        self.btnConnect.clicked.connect(self._connect)
        self.btnDisconnect.clicked.connect(self._disconnect)
        self.btnCarry.clicked.connect(self._emit_selection)
        self.btnAll.clicked.connect(lambda: self._quick_select("all"))
        self.btnNone.clicked.connect(lambda: self._quick_select("none"))
        self.btnEven.clicked.connect(lambda: self._quick_select("even"))
        self.btnOdd.clicked.connect(lambda: self._quick_select("odd"))
        self.listCh.itemChanged.connect(lambda _=None: self._emit_selection())
        self.btnEnableCross.clicked.connect(lambda: self._apply_threshold_to_checks(enable=True))
        self.btnDisableCross.clicked.connect(lambda: self._apply_threshold_to_checks(enable=False))

        # toggle LSL/Device rows
        def _toggle_rows(i):
            is_lsl = (i == 0)
            self._wLSL.setVisible(is_lsl)
            self._wDev.setVisible(not is_lsl)

        self.cmbSource.currentIndexChanged.connect(_toggle_rows)
        _toggle_rows(self.cmbSource.currentIndex())

        self._on_mode_changed(0)

    # ---------- helpers ----------
    def _rebuild_channel_list(self, n):
        self.listCh.blockSignals(True)
        self.listCh.clear()
        for i in range(n):
            it = QtWidgets.QListWidgetItem(f"ch{i:02d}")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.listCh.addItem(it)
        self.listCh.blockSignals(False)

    def _on_chcount_changed(self, n):
        self._rebuild_channel_list(n)
        # resize bar plot
        self._barPlot.clear()
        self._barBars = None

    def _on_mode_changed(self, idx):
        self.stack.setCurrentIndex(idx)

    def _quick_select(self, mode):
        n = self.spnCh.value()
        for i in range(n):
            it = self.listCh.item(i)
            if mode == "all":  st = QtCore.Qt.Checked
            elif mode == "none": st = QtCore.Qt.Unchecked
            elif mode == "even": st = QtCore.Qt.Checked if (i % 2 == 0) else QtCore.Qt.Unchecked
            else:                st = QtCore.Qt.Checked if (i % 2 == 1) else QtCore.Qt.Unchecked
            it.setCheckState(st)
        self._emit_selection()

    def _selected_indices(self):
        n = self.spnCh.value()
        return [i for i in range(n) if self.listCh.item(i).checkState() == QtCore.Qt.Checked]

    def _emit_selection(self):
        self.selectedChannelsChanged.emit(self._selected_indices())

    # ---------- LSL ----------
    def _connect(self):
        src = self.cmbSource.currentText()

        try:
            if src == "LSL stream":
                # Subscribe only
                self._rhx = None
                self.client = LSLClient(
                    stream_name=self.leName.text().strip() or None,
                    stream_type=self.leType.text().strip() or "EMG",
                    channels_hint=int(self.spnCh.value()),
                )
                self.client.start_streaming()

            else:  # Intan device (direct)
                self.client = None
                host = self.leHost.text().strip() or "127.0.0.1"
                cmd = int(self.leCmd.text().strip() or "5000")
                dat = int(self.leData.text().strip() or "5001")

                # Direct device usage (no LSL needed here)
                self._rhx = IntanRHXDevice(
                    host=host,
                    command_port=cmd,
                    data_port=dat,
                    num_channels=int(self.spnCh.value()),
                    sample_rate=None,  # let device tell us
                    buffer_duration_sec=max(5.0, float(self.dsbWin.value()) * 4),
                    auto_start=True,
                    use_lsl=False,  # direct mode
                    lsl_options=None,
                    verbose=False,
                )
                # Start streaming if not already auto-started
                if not getattr(self._rhx, "streaming", False):
                    self._rhx.start_streaming()

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Connect failed", str(e))
            self.client = None
            self._rhx = None
            return

        self.btnConnect.setEnabled(False)
        self.btnDisconnect.setEnabled(True)
        self.timer.start(int(1000 / self.ui_hz))

    def _disconnect(self):
        self.timer.stop()
        # LSL subscriber
        try:
            if self.client:
                self.client.stop()
        except Exception:
            pass
        self.client = None

        # Device
        if self._rhx:
            try:
                self._rhx.stop_streaming()
                self._rhx.close()
            except Exception:
                pass
            self._rhx = None

        self.btnConnect.setEnabled(True)
        self.btnDisconnect.setEnabled(False)

    # ---------- draw ----------
    def _tick(self):
        win = float(self.dsbWin.value())

        # choose source
        if self.client is not None:
            # LSL subscriber path
            t_rel, Y = self.client.latest(window_secs=win)  # Y: (C, T)
            if t_rel is None or Y.size == 0:
                return

        elif self._rhx is not None:
            # Direct device path (no timestamps from Intan ring; we only need amplitude for RMS)
            # returns (channels, samples) window in µV
            try:
                Y = self._rhx.get_latest_window(int(win * 1000.0))  # ms
            except Exception:
                return
        else:
            return

        # channel selection
        idx = self._selected_indices()
        if not idx:
            return
        C = np.array(idx, dtype=int)
        Y = Y[C, :]

        # RMS per channel
        rms = np.sqrt(np.mean(Y * Y, axis=1) + 1e-12)

        # Remember for threshold actions outside draw loop
        self._last_rms = rms
        self._last_sel_idx = C  # global channel indices for current selection

        if self.stack.currentIndex() == 0:
            self._draw_bars(rms, C)
        else:
            self._draw_grid(rms, len(idx))

    # def _draw_bars(self, rms, chan_idx):
    #     x = np.arange(rms.size)
    #     if self._barBars is None:
    #         bg = pg.BarGraphItem(x=x, height=rms, width=0.8, brush=pg.mkBrush("#74A9FF"))
    #         self._barBars = bg
    #         self._barPlot.addItem(bg)
    #         self._barPlot.setLabel("bottom", "channel")
    #         self._barPlot.setLabel("left", "RMS (µV)")
    #         self._barPlot.setXRange(-1, rms.size+1, padding=0)
    #     else:
    #         self._barBars.setOpts(x=x, height=rms)
    def _draw_bars(self, rms, chan_idx):
        x = np.arange(rms.size, dtype=float)
        thr = float(self.dsbThresh.value())

        lo_mask = rms <= thr
        hi_mask = ~lo_mask

        # --- low (≤ threshold) series ---
        if self._barBarsLo is None:
            bg_lo = pg.BarGraphItem(
                x=x[lo_mask],
                height=rms[lo_mask],
                width=0.8,
                brush=pg.mkBrush("#74A9FF"),
            )
            self._barBarsLo = bg_lo
            self._barPlot.addItem(bg_lo)
            self._barPlot.setLabel("bottom", "channel")
            self._barPlot.setLabel("left", "RMS (µV)")
            self._barPlot.setXRange(-1, rms.size + 1, padding=0)
        else:
            self._barBarsLo.setOpts(x=x[lo_mask], height=rms[lo_mask])

        # --- high (> threshold) series ---
        if self._barBarsHi is None:
            bg_hi = pg.BarGraphItem(
                x=x[hi_mask],
                height=rms[hi_mask],
                width=0.8,
                brush=pg.mkBrush("#FF6B6B"),
            )
            self._barBarsHi = bg_hi
            self._barPlot.addItem(bg_hi)
        else:
            self._barBarsHi.setOpts(x=x[hi_mask], height=rms[hi_mask])

        ymax = float(np.percentile(rms, 99.0)) * 1.1 + 1e-6
        self._barPlot.setYRange(0, ymax)


    def _draw_grid(self, rms, nsel):
        # build approximately square grid that fits all selected channels
        cols = int(np.ceil(np.sqrt(nsel)))
        rows = int(np.ceil(nsel / max(1, cols)))
        A = np.zeros((rows, cols), dtype=np.float32)
        flat = rms.astype(np.float32)
        A.flat[:flat.size] = flat  # pad rest with 0
        self._imgItem.setImage(A.T, autoLevels=True)  # transpose just for nicer aspect

    def _apply_threshold_to_checks(self, enable: bool):
        """
        enable=True  -> check channels with RMS > threshold
        enable=False -> uncheck channels with RMS > threshold
        Uses the most recent window and current selection mapping.
        """
        if self._last_rms is None or self._last_sel_idx is None:
            return

        thr = float(self.dsbThresh.value())
        over = (self._last_rms > thr)

        # Map local index in the selection → global channel id used by the checkbox list
        for local_i, is_over in enumerate(over):
            if not is_over:
                continue
            global_ch = int(self._last_sel_idx[local_i])
            if 0 <= global_ch < self.spnCh.value():
                it = self.listCh.item(global_ch)
                if it is not None:
                    it.setCheckState(QtCore.Qt.Checked if enable else QtCore.Qt.Unchecked)

        # propagate selection change
        self._emit_selection()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1150, 760)

        # Channel selection state (shared across tabs)
        self._selected_channels = []  # <-- initialize so profile save/load is safe
        self.n_channels = 128
        self.channels_enabled = None  # np.bool_ array length n_channels

        self.scripts = default_scripts()
        self.runner  = ProcRunner(self)
        self.runner.started.connect(self._on_started)
        self.runner.line.connect(self.append_log)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self.append_log)

        # ---------- Central layout with a vertical splitter ----------
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central); vbox.setContentsMargins(8, 8, 8, 8); vbox.setSpacing(8)

        # Shared settings "card"
        #self.top_card = self._build_shared_header()
        #self.top_card.setObjectName("Card")
        #vbox.addWidget(self.top_card)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setIconSize(QtCore.QSize(20, 20))
        vbox.addWidget(self.tabs, 1)

        self.tab_scripts = self._build_tab_scripts()
        self.tab_channel = ChannelViewWidget()
        self.tab_channel.selectedChannelsChanged.connect(self._on_channel_selection)
        self.tab_dataset = self._build_tab_dataset()
        self.tab_train   = self._build_tab_train()
        self.tab_predict = self._build_tab_predict()
        self.tab_live    = self._build_tab_live()

        # Decide icon color based on current palette (dark vs light)
        is_dark = self._is_dark()
        icolor = SVG_COLOR_DARK if is_dark else SVG_COLOR_LIGHT

        self.tabs.addTab(self.tab_scripts, _icon(svg_code(icolor)), "Scripts")
        self.tabs.addTab(self.tab_channel, _icon(svg_rms(icolor)), "Channel View")
        self.tabs.addTab(self.tab_dataset, _icon(svg_folder(icolor)), "Dataset")
        self.tabs.addTab(self.tab_train, _icon(svg_gear(icolor)), "Train")
        self.tabs.addTab(self.tab_predict, _icon(svg_play(icolor)), "Predict")
        self.tabs.addTab(self.tab_live, _icon(svg_live(icolor)), "Live")


        # Log pane inside splitter
        self.log = QtWidgets.QPlainTextEdit(objectName="LogPane")
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.log.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.log.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.log.setMaximumBlockCount(10000)
        self.log.setPlaceholderText("Logs will appear here…")
        self.log.setStyleSheet("""
            QPlainTextEdit#LogPane {
                background:#0D0D0D; color:#D1D5DB; border:0; border-radius:10px;
                font-family: Consolas, Menlo, monospace; font-size: 12px;
            }
        """)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(self.log)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)

        # Nice initial ratio (updates itself on window resizes)
        #QtCore.QTimer.singleShot(0, lambda: self.splitter.setSizes([
        #    int(self.height() * 0.58),  # tabs
        #    int(self.height() * 0.42),  # log
        #]))
        QtCore.QTimer.singleShot(0, lambda: self.splitter.setSizes([
            int(self.height() * 0.75),  # tabs (taller)
            int(self.height() * 0.25),  # log
        ]))
        vbox.addWidget(self.splitter, 1)

        # Toolbar + status bar
        self._build_toolbar()
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        self.status_msg = QtWidgets.QLabel("Ready"); self.status.addPermanentWidget(self.status_msg, 1)

        # Data
        self._load_default_profile()
        self._validate_required()

    # --------------------------- Toolbar ---------------------------
    def _build_toolbar(self):
        tb = QtWidgets.QToolBar(); tb.setMovable(False); tb.setIconSize(QtCore.QSize(20,20))
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)

        self.act_run = QtWidgets.QAction(QtGui.QIcon.fromTheme("media-playback-start"), "Run", self)
        self.act_run.setShortcut("Meta+R" if sys.platform=="darwin" else "Ctrl+R")
        self.act_run.triggered.connect(self._run_current_tab)
        tb.addAction(self.act_run)

        self.act_stop = QtWidgets.QAction(QtGui.QIcon.fromTheme("media-playback-stop"), "Stop", self)
        self.act_stop.setShortcut("Meta+." if sys.platform=="darwin" else "Ctrl+.")
        self.act_stop.triggered.connect(self.runner.stop)
        tb.addAction(self.act_stop)

        # Can choose later if wanting to keep these or use tab-specific buttons
        self.act_run.setVisible(False)
        self.act_stop.setVisible(False)

        tb.addSeparator()

        act_clear = QtWidgets.QAction(QtGui.QIcon.fromTheme("edit-clear"), "Clear Log", self)
        act_clear.setShortcut("Meta+K" if sys.platform=="darwin" else "Ctrl+K")
        act_clear.triggered.connect(lambda: self.log.setPlainText(""))
        tb.addAction(act_clear)

        tb.addSeparator()

        act_load = QtWidgets.QAction(QtGui.QIcon.fromTheme("document-open"), "Load Profile…", self)
        act_load.setShortcut("Meta+O" if sys.platform=="darwin" else "Ctrl+O")
        act_load.triggered.connect(self._load_profile_dialog)
        tb.addAction(act_load)

        act_save = QtWidgets.QAction(QtGui.QIcon.fromTheme("document-save"), "Save Profile", self)
        act_save.setShortcut("Meta+S" if sys.platform=="darwin" else "Ctrl+S")
        act_save.triggered.connect(self._save_profile_to_root)
        tb.addAction(act_save)

        # View → Light/Dark quick toggle
        menu_view = self.menuBar().addMenu("&View")
        a_light = menu_view.addAction("Light"); a_dark = menu_view.addAction("Dark")
        a_light.triggered.connect(
            lambda: (Style.apply(QtWidgets.QApplication.instance(), dark=False), self._retint_tab_icons()))
        a_dark.triggered.connect(
            lambda: (Style.apply(QtWidgets.QApplication.instance(), dark=True), self._retint_tab_icons()))

    # --------------------------- Shared Header ---------------------
    def _build_shared_header(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        lay = QtWidgets.QGridLayout(card)
        lay.setContentsMargins(12,12,12,12); lay.setHorizontalSpacing(12); lay.setVerticalSpacing(10)

        row = 0
        self.le_python = QtWidgets.QLineEdit(which_python())
        btn_py = QtWidgets.QPushButton("Browse…"); btn_py.clicked.connect(lambda: self._choose_path(self.le_python, file_mode=True))
        lay.addWidget(QtWidgets.QLabel("Python"), row, 0); lay.addWidget(self.le_python, row, 1); lay.addWidget(btn_py, row, 2); row += 1

        self.le_root = QtWidgets.QLineEdit()
        btn_root = QtWidgets.QPushButton("Browse…"); btn_root.clicked.connect(lambda: self._choose_path(self.le_root, dir_mode=True))
        lay.addWidget(QtWidgets.QLabel("Root folder"), row, 0); lay.addWidget(self.le_root, row, 1); lay.addWidget(btn_root, row, 2); row += 1

        self.le_label = QtWidgets.QLineEdit("gestures")
        self.chk_global_verbose = QtWidgets.QCheckBox("--verbose")
        h = QtWidgets.QHBoxLayout(); w = QtWidgets.QWidget(); w.setLayout(h)
        h.addWidget(self.le_label); h.addStretch(1); h.addWidget(self.chk_global_verbose)
        lay.addWidget(QtWidgets.QLabel("Dataset label"), row, 0); lay.addWidget(w, row, 1, 1, 2); row += 1

        self.le_build_script = QtWidgets.QLineEdit(self.scripts["build"])
        btn_build = QtWidgets.QPushButton("…"); btn_build.clicked.connect(lambda: self._choose_path(self.le_build_script, file_mode=True))
        lay.addWidget(QtWidgets.QLabel("Build script"), row, 0); lay.addWidget(self.le_build_script, row, 1); lay.addWidget(btn_build, row, 2); row += 1

        self.le_train_script = QtWidgets.QLineEdit(self.scripts["train"])
        btn_train = QtWidgets.QPushButton("…"); btn_train.clicked.connect(lambda: self._choose_path(self.le_train_script, file_mode=True))
        lay.addWidget(QtWidgets.QLabel("Train script"), row, 0); lay.addWidget(self.le_train_script, row, 1); lay.addWidget(btn_train, row, 2); row += 1

        self.le_rt_script = QtWidgets.QLineEdit(self.scripts["realtime"])
        btn_rt = QtWidgets.QPushButton("…"); btn_rt.clicked.connect(lambda: self._choose_path(self.le_rt_script, file_mode=True))
        lay.addWidget(QtWidgets.QLabel("Live script"), row, 0); lay.addWidget(self.le_rt_script, row, 1); lay.addWidget(btn_rt, row, 2)

        # reactive validation
        for w in (self.le_root, self.le_python):
            w.textChanged.connect(self._validate_required)
        return card

    # --------------------------- Tabs --------------------------------
    def _build_tab_scripts(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        row = 0
        self.le_python = QtWidgets.QLineEdit(which_python())
        btn_py = QtWidgets.QPushButton("Browse…")
        btn_py.clicked.connect(lambda: self._choose_path(self.le_python, file_mode=True))
        grid.addWidget(QtWidgets.QLabel("Python"), row, 0)
        grid.addWidget(self.le_python, row, 1)
        grid.addWidget(btn_py, row, 2)
        row += 1

        self.le_root = QtWidgets.QLineEdit()
        btn_root = QtWidgets.QPushButton("Browse…")
        btn_root.clicked.connect(lambda: self._choose_path(self.le_root, dir_mode=True))
        grid.addWidget(QtWidgets.QLabel("Root folder"), row, 0)
        grid.addWidget(self.le_root, row, 1)
        grid.addWidget(btn_root, row, 2)
        row += 1

        self.le_label = QtWidgets.QLineEdit("gestures")
        self.chk_global_verbose = QtWidgets.QCheckBox("--verbose")
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.le_label)
        h.addStretch(1)
        h.addWidget(self.chk_global_verbose)
        wrap = QtWidgets.QWidget();
        wrap.setLayout(h)
        grid.addWidget(QtWidgets.QLabel("Dataset label"), row, 0)
        grid.addWidget(wrap, row, 1, 1, 2)
        row += 1

        self.le_build_script = QtWidgets.QLineEdit(self.scripts["build"])
        btn_build = QtWidgets.QPushButton("…")
        btn_build.clicked.connect(lambda: self._choose_path(self.le_build_script, file_mode=True))
        grid.addWidget(QtWidgets.QLabel("Build script"), row, 0)
        grid.addWidget(self.le_build_script, row, 1)
        grid.addWidget(btn_build, row, 2)
        row += 1

        self.le_train_script = QtWidgets.QLineEdit(self.scripts["train"])
        btn_train = QtWidgets.QPushButton("…")
        btn_train.clicked.connect(lambda: self._choose_path(self.le_train_script, file_mode=True))
        grid.addWidget(QtWidgets.QLabel("Train script"), row, 0)
        grid.addWidget(self.le_train_script, row, 1)
        grid.addWidget(btn_train, row, 2)
        row += 1

        self.le_rt_script = QtWidgets.QLineEdit(self.scripts["realtime"])
        btn_rt = QtWidgets.QPushButton("…")
        btn_rt.clicked.connect(lambda: self._choose_path(self.le_rt_script, file_mode=True))
        grid.addWidget(QtWidgets.QLabel("Live script"), row, 0)
        grid.addWidget(self.le_rt_script, row, 1)
        grid.addWidget(btn_rt, row, 2)

        # validate when paths change
        for w_ in (self.le_root, self.le_python):
            w_.textChanged.connect(self._validate_required)

        return w

    def _build_tab_dataset(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(w)
        form.setHorizontalSpacing(12); form.setVerticalSpacing(10)

        self.ds_file_path = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton("Browse…"); b.clicked.connect(lambda: self._choose_path(self.ds_file_path, file_mode=True))
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.ds_file_path); h.addWidget(b); row = QtWidgets.QWidget(); row.setLayout(h)
        form.addRow("Recording file/folder", row)

        self.ds_events = QtWidgets.QLineEdit("{label}_emg.event")
        form.addRow("Events file pattern", self.ds_events)

        self.ds_overwrite = QtWidgets.QCheckBox("--overwrite")
        self.ds_verbose   = QtWidgets.QCheckBox("--verbose")
        opt = QtWidgets.QHBoxLayout(); opt.addWidget(self.ds_overwrite); opt.addWidget(self.ds_verbose); opt.addStretch(1)
        holder = QtWidgets.QWidget(); holder.setLayout(opt)
        form.addRow("Options", holder)

        self.ds_use_selected = QtWidgets.QCheckBox("Use channels selected in Channel View")
        self.ds_use_selected.setChecked(False)
        self.ds_selected_preview = QtWidgets.QLabel("(none)");
        self.ds_selected_preview.setProperty("hint", True)
        form.addRow(self.ds_use_selected)
        form.addRow("Selected channels", self.ds_selected_preview)

        self.btn_build = QtWidgets.QPushButton("Build Training Dataset");
        self.btn_build.setProperty("accent", True)
        self.btn_build.clicked.connect(self._on_build_clicked)
        form.addRow(self.btn_build)
        return w

    def _build_tab_train(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(w)
        form.setHorizontalSpacing(12); form.setVerticalSpacing(10)

        self.tr_dataset_name = QtWidgets.QLineEdit("{label}_training_dataset.npz")
        form.addRow("Dataset file name", self.tr_dataset_name)

        self.tr_save_eval  = QtWidgets.QCheckBox("--save_eval")
        self.tr_overwrite  = QtWidgets.QCheckBox("--overwrite")
        self.tr_verbose    = QtWidgets.QCheckBox("--verbose")
        opt = QtWidgets.QHBoxLayout(); [opt.addWidget(x) for x in (self.tr_save_eval, self.tr_overwrite, self.tr_verbose)]
        opt.addStretch(1); holder = QtWidgets.QWidget(); holder.setLayout(opt)
        form.addRow("Options", holder)

        self.btn_train = QtWidgets.QPushButton("Train Model"); self.btn_train.setProperty("accent", True)
        self.btn_train.clicked.connect(self._on_train_clicked)
        form.addRow(self.btn_train)
        return w

    def _build_tab_predict(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(w)
        form.setHorizontalSpacing(12); form.setVerticalSpacing(10)

        self.pr_file = QtWidgets.QLineEdit()
        b1 = QtWidgets.QPushButton("Browse…"); b1.clicked.connect(lambda: self._choose_path(self.pr_file, file_mode=True))
        r1 = QtWidgets.QHBoxLayout(); r1.addWidget(self.pr_file); r1.addWidget(b1); wrap1 = QtWidgets.QWidget(); wrap1.setLayout(r1)
        form.addRow("Prediction file path", wrap1)

        self.pr_events = QtWidgets.QLineEdit("{label}_emg.event")
        b2 = QtWidgets.QPushButton("Browse…"); b2.clicked.connect(lambda: self._choose_path(self.pr_events, file_mode=True))
        r2 = QtWidgets.QHBoxLayout(); r2.addWidget(self.pr_events); r2.addWidget(b2); wrap2 = QtWidgets.QWidget(); wrap2.setLayout(r2)
        form.addRow("Events file (for eval)", wrap2)

        self.pr_use_lsl = QtWidgets.QCheckBox("--use_lsl"); self.pr_use_lsl.setChecked(True)
        form.addRow("Options", self.pr_use_lsl)

        self.btn_predict = QtWidgets.QPushButton("Run Prediction"); self.btn_predict.setProperty("accent", True)
        self.btn_predict.clicked.connect(self._on_predict_clicked)
        form.addRow(self.btn_predict)
        return w

    def _build_tab_live(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        # ------- fields -------
        self.live_host = QtWidgets.QLineEdit("127.0.0.1")
        self.live_port = QtWidgets.QLineEdit("6000")
        self.live_window_ms = QtWidgets.QLineEdit("")
        self.live_step_ms = QtWidgets.QLineEdit("")
        self.live_infer_ms = QtWidgets.QLineEdit("")
        self.live_infer_hz = QtWidgets.QLineEdit("")
        self.live_smooth_k = QtWidgets.QLineEdit("5")
        self.live_seconds = QtWidgets.QLineEdit("0")
        self.live_use_lsl = QtWidgets.QCheckBox("--use_lsl");
        self.live_use_lsl.setChecked(False)

        # Small helpers to keep the three-across layout tidy
        c = 0  # column tracker
        r = 0  # row tracker

        def _cell(title: str, widget: QtWidgets.QWidget, maxw: int = 300) -> QtWidgets.QWidget:
            """Small card-like cell with a tiny caption above the field."""
            wrap = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(wrap)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            lbl = QtWidgets.QLabel(title)
            lbl.setProperty("hint", True)  # picks up the subtle gray from your stylesheet
            v.addWidget(lbl)
            widget.setMaximumWidth(maxw)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            v.addWidget(widget)
            return wrap

        # Row 0 (connection + one spare slot)
        grid.addWidget(_cell("Intan host", self.live_host), r, 0, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(_cell("Port", self.live_port, maxw=140), r, 1, alignment=QtCore.Qt.AlignLeft)
        grid.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum), r, 2)
        r += 1

        # Row 1 (three across)
        grid.addWidget(_cell("window_ms (override)", self.live_window_ms), r, 0, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(_cell("step_ms (override)", self.live_step_ms), r, 1, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(_cell("infer_ms (alt)", self.live_infer_ms), r, 2, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Row 2 (three across)
        grid.addWidget(_cell("infer_hz (alt)", self.live_infer_hz), r, 0, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(_cell("smooth_k", self.live_smooth_k), r, 1, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(_cell("seconds_total (0=∞)", self.live_seconds), r, 2, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Row 3 (options + status)
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.setContentsMargins(0, 0, 0, 0)
        opt_row.setSpacing(10)
        opt_row.addWidget(self.live_use_lsl)
        opt_row.addStretch(1)
        opt_wrap = QtWidgets.QWidget();
        opt_wrap.setLayout(opt_row)
        grid.addWidget(opt_wrap, r, 0, 1, 1, alignment=QtCore.Qt.AlignLeft)

        self.lbl_current = QtWidgets.QLabel("current: —")
        self.lbl_conf = QtWidgets.QLabel("p≈—")
        self.lbl_time = QtWidgets.QLabel("t=0.00s")
        status = QtWidgets.QHBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(12)
        status.addWidget(self.lbl_current)
        status.addWidget(self.lbl_conf)
        status.addWidget(self.lbl_time)
        status.addStretch(1)
        status_wrap = QtWidgets.QWidget()
        status_wrap.setLayout(status)
        grid.addWidget(status_wrap, r, 1, 1, 2)  # span two columns to the right
        r += 1

        # Row 4 (buttons)
        btns = QtWidgets.QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(12)
        self.btn_stream_start = QtWidgets.QPushButton("Start Streaming");
        self.btn_stream_start.setProperty("accent", True)
        self.btn_stream_stop = QtWidgets.QPushButton("Stop");
        self.btn_stream_stop.setEnabled(False)
        btns.addWidget(self.btn_stream_start, 1)
        btns.addWidget(self.btn_stream_stop, 1)
        btns_wrap = QtWidgets.QWidget();
        btns_wrap.setLayout(btns)
        grid.addWidget(btns_wrap, r, 0, 1, 3)

        # Make three columns share leftover width but keep cells compact
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        # Wire up
        self.btn_stream_start.clicked.connect(self._on_stream_start)
        self.btn_stream_stop.clicked.connect(self._on_stream_stop)
        return w

    def _cell(self, title: str, widget: QtWidgets.QWidget, maxw: int = 300) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap);
        v.setContentsMargins(0, 0, 0, 0);
        v.setSpacing(4)
        lbl = QtWidgets.QLabel(title);
        lbl.setProperty("hint", True);
        v.addWidget(lbl)
        widget.setMaximumWidth(maxw)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        v.addWidget(widget)
        return wrap

    def _update_ds_channels_label(self):
        if self.channels_enabled is None:
            self.lbl_ds_channels.setText("Channels: (inherit from Channel View)")
        else:
            k = int(np.sum(self.channels_enabled))
            self.lbl_ds_channels.setText(f"Channels selected: {k}/{self.n_channels}")

    def _is_dark(self):
        return self.palette().color(QtGui.QPalette.Window).value() < 128

    # --------------------------- Actions / IO -----------------------------
    def _choose_path(self, line_edit: QtWidgets.QLineEdit, *, file_mode=False, dir_mode=False):
        if file_mode:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", str(Path.cwd()))
        elif dir_mode:
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory", str(Path.cwd()))
        else:
            p = ""
        if p:
            line_edit.setText(p)

    def _compose_common(self):
        args = ["--root_dir", self.le_root.text().strip(), "--label", self.le_label.text().strip()]
        if self.chk_global_verbose.isChecked():
            args.append("--verbose")
        return args

    def _on_channel_selection(self, idx_list: list):
        self._selected_channels = list(map(int, idx_list))
        # compact preview
        if not self._selected_channels:
            txt = "(none)"
        elif len(self._selected_channels) <= 24:
            txt = ", ".join(map(str, self._selected_channels))
        else:
            head = ", ".join(map(str, self._selected_channels[:18]))
            tail = ", ".join(map(str, self._selected_channels[-4:]))
            txt = f"{head}, …, {tail}  ({len(self._selected_channels)} ch)"
        self.ds_selected_preview.setText(txt)

    def _on_build_clicked(self):
        py = self.le_python.text().strip() or which_python()
        script = self.le_build_script.text().strip()
        args = [py, script] + self._compose_common()

        if (fp := self.ds_file_path.text().strip()):
            args += ["--file_dir", fp]

        ev = self.ds_events.text().strip().format(label=self.le_label.text().strip())
        if not os.path.isabs(ev):
            ev = os.path.join(self.le_root.text().strip(), "events", ev)
        args += ["--events_file", ev]

        if self.ds_overwrite.isChecked():
            args.append("--overwrite")
        if self.ds_verbose.isChecked():
            args.append("--verbose")

        # Carry channel selection into dataset build if requested on Channel View
        if self.ds_use_selected.isChecked() and getattr(self, "_selected_channels", None):
            # Prefer --channels if your script supports it; else uncomment the file path path.
            args += ["--channels", ",".join(map(str, self._selected_channels))]

        self.runner.run(args)

    def _on_train_clicked(self):
        py = self.le_python.text().strip() or which_python()
        script = self.le_train_script.text().strip()
        args = [py, script] + self._compose_common()

        name = self.tr_dataset_name.text().strip().format(label=self.le_label.text().strip())
        args += ["--dataset_dir", name]
        if self.tr_save_eval.isChecked(): args.append("--save_eval")
        if self.tr_overwrite.isChecked(): args.append("--overwrite")
        if self.tr_verbose.isChecked():   args.append("--verbose")
        self.runner.run(args)

    def _on_predict_clicked(self):
        py = self.le_python.text().strip() or which_python()
        script = self.le_rt_script.text().strip()
        args = [py, script] + self._compose_common()

        if (pf := self.pr_file.text().strip()):
            args += ["--file_dir", pf]
        ev = self.pr_events.text().strip().format(label=self.le_label.text().strip())
        if not os.path.isabs(ev):
            ev = os.path.join(self.le_root.text().strip(), "events", ev)
        args += ["--events_file", ev]
        if self.pr_use_lsl.isChecked():
            args.append("--use_lsl")
        self.runner.run(args)

    def _on_stream_start(self):
        py = self.le_python.text().strip() or which_python()
        script = self.le_rt_script.text().strip()
        args = [py, script] + self._compose_common()

        def add(flag, w):
            t = w.text().strip()
            if t: args.extend([flag, t])

        add("--window_ms", self.live_window_ms)
        add("--step_ms",   self.live_step_ms)
        add("--infer_ms",  self.live_infer_ms)
        add("--infer_hz",  self.live_infer_hz)
        add("--smooth_k",  self.live_smooth_k)
        add("--seconds",   self.live_seconds)
        if self.live_use_lsl.isChecked():
            args.append("--use_lsl")

        self.btn_stream_start.setEnabled(False)
        self.btn_stream_stop.setEnabled(True)
        self.runner.run(args)

    def _on_stream_stop(self):
        self.runner.stop()
        self.btn_stream_stop.setEnabled(False)
        self.btn_stream_start.setEnabled(True)

    def _rebuild_channel_table(self, n):
        self.n_channels = n
        rows, cols = (8, 8) if n <= 64 else (8, 16)
        self.cv_table.clear()
        self.cv_table.setRowCount(rows);
        self.cv_table.setColumnCount(cols)
        self.cv_table.horizontalHeader().setVisible(False);
        self.cv_table.verticalHeader().setVisible(False)
        self.cv_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cv_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < n:
                    it = QtWidgets.QTableWidgetItem(f"{idx}")
                    it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                    checked = True if (self.channels_enabled is None or (
                                idx < len(self.channels_enabled) and self.channels_enabled[idx])) else False
                    it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
                    self.cv_table.setItem(r, c, it)
                    idx += 1
                else:
                    it = QtWidgets.QTableWidgetItem("")
                    it.setFlags(QtCore.Qt.NoItemFlags)
                    self.cv_table.setItem(r, c, it)

    # --------------------------- Channel View logic -------------------------
    # def _cv_get_mask(self) -> np.ndarray:
    #     n = self.n_channels
    #     rows, cols = (8, 8) if n <= 64 else (8, 16)
    #     mask = np.zeros(n, dtype=bool)
    #     i = 0
    #     for r in range(rows):
    #         for c in range(cols):
    #             it = self.cv_table.item(r, c)
    #             if it and (it.flags() & QtCore.Qt.ItemIsUserCheckable):
    #                 mask[i] = (it.checkState() == QtCore.Qt.Checked)
    #                 i += 1
    #     return mask
    #
    # def _cv_set_all(self, state: bool):
    #     n = self.n_channels
    #     rows, cols = (8, 8) if n <= 64 else (8, 16)
    #     for r in range(rows):
    #         for c in range(cols):
    #             it = self.cv_table.item(r, c)
    #             if it and (it.flags() & QtCore.Qt.ItemIsUserCheckable):
    #                 it.setCheckState(QtCore.Qt.Checked if state else QtCore.Qt.Unchecked)
    #
    # def _cv_invert(self):
    #     n = self.n_channels
    #     rows, cols = (8, 8) if n <= 64 else (8, 16)
    #     for r in range(rows):
    #         for c in range(cols):
    #             it = self.cv_table.item(r, c)
    #             if it and (it.flags() & QtCore.Qt.ItemIsUserCheckable):
    #                 it.setCheckState(QtCore.Qt.Unchecked if it.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked)
    #
    # def _cv_switch_view(self, mode: str):
    #     if mode == "Bar":
    #         self.cv_stack.setCurrentIndex(0)
    #         self.cv_heat.setVisible(False);
    #         self.cv_barplot.setVisible(True)
    #     else:
    #         self.cv_stack.setCurrentIndex(1)
    #         self.cv_barplot.setVisible(False);
    #         self.cv_heat.setVisible(True)
    #         self.cv_heat.view.setAspectLocked(True)
    #
    # def _cv_start(self):
    #     if hasattr(self, "_rms_worker") and self._rms_worker is not None:
    #         return
    #     self.channels_enabled = self._cv_get_mask()
    #     self._rms_worker = RMSWorker(
    #         n_channels=int(self.cv_nch.currentText()),
    #         fs_hint=2000.0,
    #         window_ms=int(self.cv_window.value()),
    #         use_lsl=self.cv_use_lsl.isChecked(),
    #         parent=self
    #     )
    #     self._rms_worker.rms_ready.connect(self._cv_update)
    #     self._rms_worker.fs_ready.connect(lambda fs: self.append_log(f"[ChannelView] fs={fs:.1f} Hz"))
    #     self._rms_worker.start()
    #     self.btn_cv_start.setEnabled(False);
    #     self.btn_cv_stop.setEnabled(True)
    #
    # def _cv_stop(self):
    #     if getattr(self, "_rms_worker", None):
    #         self._rms_worker.stop();
    #         self._rms_worker.wait(800);
    #         self._rms_worker = None
    #     self.btn_cv_start.setEnabled(True);
    #     self.btn_cv_stop.setEnabled(False)
    #
    # def _cv_update(self, rms: np.ndarray):
    #     mask = self._cv_get_mask()
    #     self.channels_enabled = mask
    #     data = rms.copy()
    #     if self.cv_norm.isChecked() and data.max() > 0:
    #         data = data / data.max()
    #
    #     if self.cv_view.currentText() == "Bar":
    #         x = np.arange(data.size)
    #         shown = data.copy();
    #         shown[~mask] *= 0.15
    #         self._cv_bar.setOpts(x=x, height=shown, width=0.85)
    #         self.cv_barplot.setXRange(-1, data.size)
    #         self.cv_barplot.setYRange(0, float(np.percentile(shown, 99.0)) * 1.1 + 1e-6)
    #     else:
    #         rows, cols = (8, 8) if self.cv_view.currentText() == "8×8 Grid" else (8, 16)
    #         grid = np.zeros((rows, cols), dtype=np.float32)
    #         n = min(data.size, rows * cols)
    #         grid.flat[:n] = data[:n]
    #         mgrid = np.zeros_like(grid) + 1.0
    #         mgrid.flat[:n] = mask[:n].astype(float)
    #         img = grid * (0.15 + 0.85 * mgrid)
    #         self.cv_heat.setImage(img.T[::-1, :], autoLevels=True, autoRange=False, autoHistogramRange=True)

    # --------------------------- Runner callbacks -------------------------
    def _on_started(self, cmd_line: str):
        self.append_log(cmd_line)
        self._set_running(True, cmd_line)

    def _on_finished(self, code: int):
        self.append_log(f"\n[process exited with code {code}]\n")
        self._set_running(False, "")

    def append_log(self, text: str):
        if text.startswith("ERROR") or "Traceback" in text:
            text = " X " + text
        elif text.startswith("WARNING") or "warn" in text.lower():
            text = " ! " + text

        # quick parse for "pred=" lines (keep your format if different)
        # [ 12.34s ] pred=Foo (p≈0.93) smoothed=Bar
        if "smoothed=" in text and "pred=" in text:
            try:
                secs = float(text.split("[")[1].split("s]")[0])
                self.lbl_time.setText(f"t={secs:.2f}s")
            except Exception:
                pass
            if "smoothed=" in text:
                self.lbl_current.setText("current: " + text.split("smoothed=")[1].split()[0])
            if "p≈" in text:
                p = text.split("p≈")[1].split(")")[0]
                self.lbl_conf.setText(f"p≈{p}")

        self.log.appendPlainText(text)
        c = self.log.textCursor()
        c.movePosition(QtGui.QTextCursor.End); self.log.setTextCursor(c)

    def _set_running(self, running: bool, label: str = ""):
        self.act_run.setEnabled(not running)
        self.act_stop.setEnabled(running)
        self.status_msg.setText("Running: " + label if running else "Ready")

    def _retint_tab_icons(self):
        is_dark = self.palette().color(QtGui.QPalette.Window).value() < 128
        color = SVG_COLOR_DARK if is_dark else SVG_COLOR_LIGHT
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_scripts), _icon(svg_code(color)))
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_channel), _icon(svg_rms(color)))
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_dataset), _icon(svg_folder(color)))
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_train), _icon(svg_gear(color)))
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_predict), _icon(svg_play(color)))
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_live), _icon(svg_live(color)))
        self.tabs.setTabIcon(self.tabs.indexOf(self.tab_channel), _icon(svg_rms(color)))

    # --------------------------- Profiles ---------------------------------
    def _profile_path_default(self) -> Path:
        return Path(__file__).with_name(DEFAULT_PROFILE_NAME)

    def _save_profile_to_root(self):
        rd = self.le_root.text().strip()
        path = (Path(rd) / DEFAULT_PROFILE_NAME) if rd else self._profile_path_default()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._current_profile(), f, indent=2)
            self.append_log(f"Saved profile to: {path}")
        except Exception as e:
            self.append_log(f"Failed to save profile: {e}")

    def _load_profile_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load profile JSON", str(Path.cwd()), "JSON (*.json)")
        if not p: return
        try:
            with open(p, "r", encoding="utf-8") as f:
                prof = json.load(f)
            self._apply_profile(prof)
            self.append_log(f"Loaded profile: {p}")
        except Exception as e:
            self.append_log(f"Failed to load profile: {e}")

    def _load_default_profile(self):
        path = self._profile_path_default()
        if path.exists():
            try:
                prof = json.load(open(path, "r", encoding="utf-8"))
                self._apply_profile(prof)
                self.append_log(f"Loaded default profile: {path}")
            except Exception:
                pass

    def _current_profile(self) -> dict:
        return {
            "python": self.le_python.text(),
            "root_dir": self.le_root.text(),
            "label": self.le_label.text(),
            "verbose": self.chk_global_verbose.isChecked(),
            "scripts": {
                "build": self.le_build_script.text(),
                "train": self.le_train_script.text(),
                "realtime": self.le_rt_script.text(),
            },
            "channel_view": {
                "lsl_type": self.tab_channel.leType.text(),
                "lsl_name": self.tab_channel.leName.text(),
                "channels": self.tab_channel.spnCh.value(),
                "rms_window": self.tab_channel.dsbWin.value(),
                "mode": self.tab_channel.cmbMode.currentText(),
                "selected": getattr(self, "_selected_channels", []),
                "host": self.tab_channel.leHost.text(),
                "port": self.tab_channel.lePort.text(),
            },
            "dataset": {
                "file_path": self.ds_file_path.text(),
                "events_pattern": self.ds_events.text(),
                "overwrite": self.ds_overwrite.isChecked(),
                "verbose": self.ds_verbose.isChecked(),
            },
            "train": {
                "dataset_name": self.tr_dataset_name.text(),
                "save_eval": self.tr_save_eval.isChecked(),
                "overwrite": self.tr_overwrite.isChecked(),
                "verbose": self.tr_verbose.isChecked(),
            },
            "prediction": {
                "file_path": self.pr_file.text(),
                "events_file": self.pr_events.text(),
                "use_lsl": self.pr_use_lsl.isChecked(),
            },
            "live": {
                "infer_ms": self.live_infer_ms.text(),
                "infer_hz": self.live_infer_hz.text(),
                "window_ms": self.live_window_ms.text(),
                "step_ms": self.live_step_ms.text(),
                "smooth_k": self.live_smooth_k.text(),
                "seconds": self.live_seconds.text(),
                "use_lsl": self.live_use_lsl.isChecked(),
                "host": self.live_host.text(),
                "port": self.live_port.text(),
            },
        }

    def _apply_profile(self, p: dict):
        self.le_python.setText(p.get("python", which_python()))
        self.le_root.setText(p.get("root_dir", ""))
        self.le_label.setText(p.get("label", "exo_gestures"))
        self.chk_global_verbose.setChecked(bool(p.get("verbose", False)))

        sc = p.get("scripts", {})
        self.le_build_script.setText(sc.get("build", self.scripts["build"]))
        self.le_train_script.setText(sc.get("train", self.scripts["train"]))
        self.le_rt_script.setText(sc.get("realtime", self.scripts["realtime"]))

        cv = p.get("channel_view", {})
        if cv:
            self.tab_channel.leType.setText(cv.get("lsl_type", "EMG"))
            self.tab_channel.leName.setText(cv.get("lsl_name", ""))
            self.tab_channel.spnCh.setValue(int(cv.get("channels", 128)))
            self.tab_channel.dsbWin.setValue(float(cv.get("rms_window", 0.25)))
            self.tab_channel.leHost.setText(cv.get("host", ""))  # NEW
            self.tab_channel.lePort.setText(str(cv.get("port", "5001")))  # NEW
            mode = cv.get("mode", "RMS Bars")
            self.tab_channel.cmbMode.setCurrentIndex(0 if mode == "RMS Bars" else 1)
            sel = cv.get("selected", list(range(self.tab_channel.spnCh.value())))
            self._on_channel_selection(sel)
            self.tab_channel._rebuild_channel_list(self.tab_channel.spnCh.value())
            for i in range(min(self.tab_channel.spnCh.value(), len(sel))):
                self.tab_channel.listCh.item(i).setCheckState(
                    QtCore.Qt.Checked if (i in sel) else QtCore.Qt.Unchecked
                )

        ds = p.get("dataset", {})
        self.ds_file_path.setText(ds.get("file_path", ""))
        self.ds_events.setText(ds.get("events_pattern", "{label}_emg.event"))
        self.ds_overwrite.setChecked(bool(ds.get("overwrite", False)))
        self.ds_verbose.setChecked(bool(ds.get("verbose", False)))

        tr = p.get("train", {})
        self.tr_dataset_name.setText(tr.get("dataset_name", "{label}_training_dataset.npz"))
        self.tr_save_eval.setChecked(bool(tr.get("save_eval", False)))
        self.tr_overwrite.setChecked(bool(tr.get("overwrite", False)))
        self.tr_verbose.setChecked(bool(tr.get("verbose", False)))

        pr = p.get("prediction", {})
        self.pr_file.setText(pr.get("file_path", ""))
        self.pr_events.setText(pr.get("events_file", "{label}_emg.event"))
        self.pr_use_lsl.setChecked(bool(pr.get("use_lsl", True)))

        lv = p.get("live", {})
        self.live_infer_ms.setText(lv.get("infer_ms", ""))
        self.live_infer_hz.setText(lv.get("infer_hz", ""))
        self.live_window_ms.setText(lv.get("window_ms", ""))
        self.live_step_ms.setText(lv.get("step_ms", ""))
        self.live_smooth_k.setText(lv.get("smooth_k", "5"))
        self.live_seconds.setText(lv.get("seconds", "0"))
        self.live_use_lsl.setChecked(bool(lv.get("use_lsl", False)))
        self.live_host.setText(lv.get("host", "127.0.0.1"))
        self.live_port.setText(str(lv.get("port", "6000")))

    # --------------------------- Validation -------------------------------
    def _validate_required(self):
        ok = bool(self.le_root.text().strip()) and bool(self.le_python.text().strip())
        self.act_run.setEnabled(ok)

    # --------------------------- Run dispatch -----------------------------
    def _run_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx == self.tabs.indexOf(self.tab_channel):
            self._cv_start()
        elif idx == self.tabs.indexOf(self.tab_dataset):
            self._on_build_clicked()
        elif idx == self.tabs.indexOf(self.tab_train):
            self._on_train_clicked()
        elif idx == self.tabs.indexOf(self.tab_predict):
            self._on_predict_clicked()
        elif idx == self.tabs.indexOf(self.tab_live):
            self._on_stream_start()

    def _stop_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx == self.tabs.indexOf(self.tab_channel):
            self._cv_stop()
        elif idx == self.tabs.indexOf(self.tab_live):
            self._on_stream_stop()
        else:
            self.runner.stop()


# ---------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    Style.apply(app, dark=False)  # switch to dark=True if you prefer
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
