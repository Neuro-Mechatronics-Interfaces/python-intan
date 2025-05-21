"""
Intan: Python interface for reading, processing, and visualizing data
from Intan Technologies' RHD2000 and RHS2000 acquisition systems.

This package includes modules for:
- Reading and parsing `.rhd` and `.rhs` files
- Streamed acquisition via TCP/IP from the Intan RHX software
- Visualization of high-density EMG or LFP signals
- Configuration and device control
"""

__version__ = "1.3.0"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/jshulgach/intan-python"
__description__ = "Python interface for streaming, parsing, and analyzing Intan Technologies RHX files"

import importlib as _importlib

submodules = [
    # 'decomposition',
    'applications',
    'io',
    'plotting',
    # 'control',
    'processing',
    'rhx_interface',
    'samples',
    # 'stream',
]

__all__ = submodules + [
    #'LowLevelCallable',
    #'tests',
    #'show_config',
    '__version__',
]


def __dir__():
    return __all__
