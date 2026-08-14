"""
Intan: Python interface for reading, processing, and visualizing data
from Intan Technologies' RHD2000 and RHS2000 acquisition systems.

This package includes modules for:
- Reading and parsing `.rhd` and `.rhs` files
- Streamed acquisition via TCP/IP from the Intan RHX software
- Visualization of high-density EMG or LFP signals
- Configuration and device control
"""

__version__ = "0.2.2"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/Neuro-Mechatronics-Interfaces/python-intan"
__description__ = "Python interface for streaming, parsing, and analyzing Intan Technologies RHX files"

from importlib import import_module as _import_module

submodules = [
    'applications',
    'decomposition',
    'io',
    'ml',
    'plotting',
    'processing',
    'interface',
    'samples',
    'ui',
]

__all__ = submodules + [
    #'LowLevelCallable',
    #'tests',
    #'show_config',
    '__version__',
]


def __dir__():
    return __all__


def __getattr__(name):
    """Import public subpackages on first access.

    This keeps ``import intan`` lightweight while supporting the documented
    ``intan.io`` and ``intan.processing`` attribute style.
    """
    if name in submodules:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
