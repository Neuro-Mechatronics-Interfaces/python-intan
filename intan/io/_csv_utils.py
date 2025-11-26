from __future__ import annotations
import os
import re
import glob
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

try:
    # Optional GUI file picker (kept consistent with other loaders)
    from tkinter import Tk, filedialog  # type: ignore
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


@dataclass
class _Guess:
    emg_cols: List[str]
    imu_cols: List[str]
    time_col: Optional[str]
    fs: Optional[float]


_EMG_PAT = re.compile(r'^(EMG[-_ ]?)(\d+)$', re.IGNORECASE)
_IMU_CANON = ['roll', 'pitch', 'yaw', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
_TIME_CANDIDATES = ['t', 'time', 'timestamp', 't_emg', 't_samples', 't_host']


def _pick_file(path: Optional[str]) -> str:
    """Return a concrete path, showing a file dialog if needed (like other loaders)."""
    if path and os.path.isfile(path):
        return path
    if path and not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    if _TK_AVAILABLE:
        root = Tk(); root.withdraw()
        f = filedialog.askopenfilename(title='Select EMG CSV',
                                       filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        root.update(); root.destroy()
        if f:
            return f
    raise FileNotFoundError("CSV path not provided or not found; and GUI picker unavailable.")


def _guess_layout(df: pd.DataFrame,
                  sample_rate_hint: Optional[float]) -> _Guess:
    cols = list(df.columns)

    # EMG columns: any EMG_x style, keep numeric sort order
    emg_cols = [c for c in cols if _EMG_PAT.match(c)]
    emg_cols_sorted = sorted(emg_cols, key=lambda c: int(_EMG_PAT.match(c).group(2)))

    # IMU columns (optional)
    imu_cols = [c for c in _IMU_CANON if c in cols]

    # Prefer an explicit time column if present
    time_col = None
    for c in _TIME_CANDIDATES:
        if c in cols:
            time_col = c
            break

    # Try to infer fs if a monotonic time column exists
    fs = None
    if time_col is not None:
        t = df[time_col].to_numpy()
        # Robust: use median dt over central portion to avoid edge junk
        if np.isfinite(t).all() and len(t) > 5:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt)]
            if dt.size:
                # guard against zeros (host timestamps could be constant)
                dt_pos = dt[dt > 0]
                if dt_pos.size:
                    zero_frac = 1.0 - (dt_pos.size / dt.size) if dt.size else 1.0
                    if zero_frac < 0.2:
                        fs = 1.0 / float(np.median(dt_pos))
                    else:
                        fs = None

    if fs is None:
        fs = sample_rate_hint

    return _Guess(emg_cols_sorted, imu_cols, time_col, fs)


def _build_amplifier_channels(names: Sequence[str]) -> List[Dict[str, Any]]:
    """Mimic the RHD loader channel dicts enough for downstream plotting helpers."""
    out = []
    for idx, name in enumerate(names):
        out.append({
            'native_channel_name': name,
            'custom_channel_name': name,
            'port_name': 'EMG',
            'port_prefix': 'EMG',
            'chip_channel': idx,
            'electrode_impedance_magnitude': None,
            'electrode_impedance_phase': None,
        })
    return out


def load_csv_file(path: Optional[str] = None,
                  sample_rate: float = 1000.0,
                  export_basename: Optional[str] = None,
                  export_basepath: Optional[str] = None,
                  return_dataframe: bool = False,
                  ) -> Dict[str, Any]:
    """
    Load a CSV containing EMG channels (EMG_0 ... EMG_N) and optional IMU columns.

    Parameters
    ----------
    path : str or None
        CSV path. If None, opens a file dialog (like other loaders).
    sample_rate : float
        Fallback EMG sampling rate (Hz) if we cannot infer from a time column.
    export_basename, export_basepath : str or None
        Included in the result to match the RHD loader contract.
    return_dataframe : bool
        If True, include the original pandas DataFrame as result['dataframe'].

    Returns
    -------
    dict
        A dictionary aligned to the structure of `intan.io.load_rhd_file`, with keys:
        - 'amplifier_data'      : np.ndarray (n_channels, n_samples) float32
        - 'amplifier_channels'  : list[dict] with 'native_channel_name', etc.
        - 'channel_names'       : list[str] of EMG column names
        - 'frequency_parameters': {'amplifier_sample_rate': float}
        - 't_amplifier'         : np.ndarray (n_samples,) seconds
        - 'board_adc_data'      : np.ndarray for IMU signals (if present) shape (n_aux, n_samples)
        - 'board_adc_channels'  : list[dict] for IMU channel descriptors
        - 'export_basename', 'export_basepath'
        - 'source_path'         : original file path
        - 'meta'                : dict with lightweight details (column maps, etc.)
        - optionally 'dataframe': the raw pandas DataFrame (if return_dataframe=True)
    """
    path = _pick_file(path)

    df = pd.read_csv(path)
    guess = _guess_layout(df, sample_rate_hint=sample_rate)
    if not guess.emg_cols:
        raise ValueError("No EMG columns found. Expected columns named like 'EMG_0', 'EMG_1', ...")

    emg = df[guess.emg_cols].to_numpy(dtype=np.float32)
    emg = np.nan_to_num(emg, nan=0.0)  # keep downstream code happy
    n_samples = emg.shape[0]

    # Time vector
    fs = float(guess.fs if guess.fs is not None else sample_rate)
    if fs <= 0:
        raise ValueError(f"Invalid/unknown sample rate: {fs}")
    if guess.time_col is not None:
        t = df[guess.time_col].to_numpy(dtype=np.float64)
        # If constant or non-monotonic host timestamps, synthesize time
        dt = np.diff(t)
        zero_frac = float(np.sum(dt==0.0))/dt.size if dt.size else 1.0
        if (not np.all(np.isfinite(dt))) or (np.nanmax(dt) == 0.0) or (zero_frac >= 0.2):
            t = np.arange(n_samples, dtype=np.float64) / fs
    else:
        t = np.arange(n_samples, dtype=np.float64) / fs

    # Optional IMU block (board_adc_*)
    adc_data = None
    adc_channels = []
    if guess.imu_cols:
        adc = df[guess.imu_cols].to_numpy(dtype=np.float32)
        adc = np.nan_to_num(adc, nan=0.0)
        # Arrange shape to match (n_channels, n_samples)
        adc_data = adc.T
        for i, name in enumerate(guess.imu_cols):
            adc_channels.append({
                'native_channel_name': name,
                'custom_channel_name': name,
                'port_name': 'IMU',
                'port_prefix': 'IMU',
                'chip_channel': i,
            })

    # Build result dictionary (align with RHD loader schema)
    src = os.path.abspath(path)
    if export_basename is None:
        export_basename = os.path.splitext(os.path.basename(src))[0]
    if export_basepath is None:
        export_basepath = os.path.dirname(src)

    result: Dict[str, Any] = {
        'source_path'         : src,
        'export_basename'     : export_basename,
        'export_basepath'     : export_basepath,
        'channel_names'       : list(guess.emg_cols),
        'amplifier_channels'  : _build_amplifier_channels(guess.emg_cols),
        'amplifier_data'      : emg.T,  # (n_channels, n_samples)
        'frequency_parameters': {'amplifier_sample_rate': fs},
        't_amplifier'         : t.astype(np.float64),
        'meta'                : {
            'csv': {
                'emg_columns': list(guess.emg_cols),
                'imu_columns': list(guess.imu_cols),
                'time_column': guess.time_col,
                'rows': int(n_samples),
            }
        }
    }

    if adc_data is not None:
        result['board_adc_data'] = adc_data
        result['board_adc_channels'] = adc_channels

    if return_dataframe:
        result['dataframe'] = df

    return result

def find_csv_dir(root: str) -> str:
    for sub in ("csv", "raw"):
        cand = os.path.join(root, sub)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"No 'csv/' or 'raw/' folder under {root}")

def load_csv_files(root_dir: str, csv_sample_rate: float = 1000.0, verbose: bool = False):
    csv_dir = os.path.join(root_dir, "csv")
    if not os.path.isdir(csv_dir):
        csv_dir = os.path.join(root_dir, "raw")
        if not os.path.isdir(csv_dir):
            return []

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    items = []
    for path in csv_paths:
        try:
            data = load_csv_file(path, sample_rate=csv_sample_rate, return_dataframe=False)
            items.append((data, path))
            if verbose:
                print(f"[csv] loaded {os.path.basename(path)}  "
                             f"(C={data['amplifier_data'].shape[0]}, N={data['amplifier_data'].shape[1]}, "
                             f"ADC={data.get('board_adc_data').shape[0] if 'board_adc_data' in data else 0}), "
                             f"fs={data['frequency_parameters']['amplifier_sample_rate']:.2f}")
        except Exception as e:
            print(f"[csv][skip] {os.path.basename(path)}: {e}")
    return items