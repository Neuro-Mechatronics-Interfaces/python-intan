#!/usr/bin/env python3
"""
Shared UI helpers for examples/applications.

Move small, commonly-used helpers (which_python, default_examples_dir, SVG helpers)
here so multiple example applications can import them without duplication.
"""
import sys
from pathlib import Path

try:
    from PyQt5 import QtSvg, QtGui, QtCore
except Exception:
    try:
        from PySide2 import QtSvg, QtGui, QtCore
    except Exception:
        QtSvg = None
        QtGui = None
        QtCore = None


def which_python() -> str:
    return sys.executable or "python"


def default_examples_dir() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parent, here.parent.parent, here.parents[2] if len(here.parents) > 2 else here.parent):
        p = base / "examples" / "gesture_classifier"
        if p.exists():
            return p
    return here.parent


def _icon(svg: str, size: int = 20):
    """Make a QIcon from a tiny inline SVG string (monochrome).

    Returns a `QtGui.QIcon` when Qt is available, otherwise returns None.
    """
    if QtSvg is None or QtGui is None or QtCore is None:
        return None
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    renderer.render(p)
    p.end()
    return QtGui.QIcon(pm)


SVG_COLOR_LIGHT = "#1C1C1E"
SVG_COLOR_DARK = "#E5E7EB"


def svg_code(color):
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <path d='M9.5 6.5 5 12l4.5 5.5' stroke='{color}' stroke-width='1.6'
              stroke-linecap='round' stroke-linejoin='round'/>
        <path d='M14.5 6.5 19 12l-4.5 5.5' stroke='{color}' stroke-width='1.6'
              stroke-linecap='round' stroke-linejoin='round'/>
    </svg>"""


def svg_folder(color):
    return f"""<svg width="24" height="24" viewBox="0 0 24 24" fill="none"
        xmlns='http://www.w3.org/2000/svg'>
        <path d="M3 6.5h5.2l1.6 2H21v8.5a2 2 0 0 1-2 2H5
                 a2 2 0 0 1-2-2V6.5Z" fill="{color}" fill-opacity="0.12"/>
        <path d="M3 7.5V6a2 2 0 0 1 2-2h3.2l1.6 2H21a2 2 0 0 1 2 2v9.5
                 a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V7.5Z"
              stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>"""


def svg_gear(color):
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <circle cx='12' cy='12' r='3.5' stroke='{color}' stroke-width='1.6'/>
        <path d='M12 2.8v2.4M12 18.8v2.4M21.2 12h-2.4M5.2 12H2.8
                 M17.5 6.5l-1.7 1.7M8.2 15.8 6.5 17.5M17.5 17.5l-1.7-1.7M8.2 8.2 6.5 6.5'
              stroke='{color}' stroke-width='1.6' stroke-linecap='round'/>
    </svg>"""


def svg_play(color):
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <path d='M8 6l10 6-10 6V6Z' fill='{color}' fill-opacity='0.14' />
        <path d='M8 6l10 6-10 6V6Z' stroke='{color}' stroke-width='1.6' stroke-linejoin='round'/>
    </svg>"""


def svg_live(color):
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <circle cx='12' cy='12' r='3.2' fill='{color}'/>
        <circle cx='12' cy='12' r='7.5' stroke='{color}' stroke-width='1.6' opacity='0.55'/>
    </svg>"""


def svg_rms(color):
    return f"""<svg width='24' height='24' viewBox='0 0 24 24' fill='none'
        xmlns='http://www.w3.org/2000/svg'>
        <rect x='5' y='10' width='2.2' height='8' fill='{color}' opacity='0.85'/>
        <rect x='9' y='6'  width='2.2' height='12' fill='{color}' opacity='0.85'/>
        <rect x='13' y='12' width='2.2' height='6' fill='{color}' opacity='0.85'/>
        <rect x='17' y='8'  width='2.2' height='10' fill='{color}' opacity='0.85'/>
        <rect x='3' y='3' width='4' height='4' stroke='{color}' stroke-width='1.4' opacity='0.7'/>
    </svg>"""
