"""
python-intan: A Python package for reading and visualizing data from Intan Technologies' RHD2000 series of integrated circuits.

Subpackages
-----------
  io                      --- Reading and writing Intan data files
  plotting                --- Plotting and visualization
  control                 --- UART and TCP communication helpers
  processing              --- Signal processing functions
"""

import importlib as _importlib

submodules = [
    'io',
    'plotting',
    'control',
    'processing',
]

__all__ = submodules + [
    'LowLevelCallable',
    'test',
    'show_config',
    '__version__',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'scipy.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'scipy' has no attribute '{name}'"
            )
