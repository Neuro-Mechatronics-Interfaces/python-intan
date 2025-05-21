"""
intan.processing._data_processing

Provides core post-processing utilities for `.rhd` and `.dat` Intan datasets.

This module converts raw binary signals into meaningful physical units (e.g., µV, V, °C),
extracts and formats digital inputs, scales timestamps, and optionally applies a notch filter.

Main functions:
- `parse_data`: Orchestrates full signal post-processing
- `scale_analog_data`: Applies voltage/temperature scaling
- `scale_timestamps`: Converts sample indices to seconds
- `extract_digital_data`: Expands digital inputs into bitwise channel rows
- `apply_notch_filter`: Applies notch filter (if required by header)

This module is typically called internally after loading data using `intan.io._rhd_loader`.
"""
import numpy as np
from intan.io._file_utils import print_progress
from ._filters import notch_filter

