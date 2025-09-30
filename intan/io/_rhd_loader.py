"""
intan.io._rhd_loader

This module provides file loaders for Intan Technologies' `.rhd` binary files,
as well as associated `.dat` files used in 'One File Per Signal Type' format.

It includes GUI-assisted selection, header parsing, signal reconstruction,
and helper functions to load and concatenate datasets. The output is a
dictionary containing signal data, metadata, and time vectors for each channel type.

Key Functions:
- `load_rhd_file`: Full `.rhd` file loader with header and data parsing
- `load_dat_file`: Loader for `.dat`-based datasets with separate header
- `load_files_from_path`: Batch loading and optional concatenation
- `read_amplifier_file`, `read_auxiliary_file`, etc.: Raw binary readers
"""

import os
import re
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
from datetime import datetime
from ._rhd_header_parsing import read_header, header_to_result, data_to_result, calculate_data_size
from ._rhd_block_parser import read_all_data_blocks
from ._file_utils import check_end_of_file, print_progress, get_file_paths
from intan.processing import notch_filter


_TS_PAT = re.compile(r'(?P<date>\d{6,8})[_-]?(?P<time>\d{6})')  # e.g. 250901_163304 or 20250901-163304



def _parse_dt_from_name(basename):
    """Try to parse YYMMDD/YYYMMDD + HHMMSS from filename stem."""
    stem = os.path.splitext(os.path.basename(basename))[0]
    m = _TS_PAT.search(stem)
    if not m:
        return None
    d, t = m.group('date'), m.group('time')
    try:
        if len(d) == 6:   # YYMMDD
            return datetime.strptime(d + t, '%y%m%d%H%M%S')
        elif len(d) == 8: # YYYYMMDD
            return datetime.strptime(d + t, '%Y%m%d%H%M%S')
    except ValueError:
        return None
    return None


def _common_prefix_no_ts(stems):
    """Longest common prefix after stripping the timestamp token."""
    cleaned = []
    for s in stems:
        s2 = _TS_PAT.sub('', s)               # remove date_time token
        s2 = s2.strip('_-. ')                 # tidy separators
        cleaned.append(s2)
    pref = os.path.commonprefix(cleaned).rstrip('_-. ')
    return pref or "merged"


def _build_export_basename(paths):
    """
    Build a clean export base name like:
    <prefix>_YYYYMMDDTHHMMSS-HHMMSS_nN_concat
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    prefix = _common_prefix_no_ts(stems)

    # parse start/end datetimes from names, else fallback to file mtimes
    dts = []
    for p in paths:
        dt = _parse_dt_from_name(os.path.basename(p))
        if dt is None:
            try:
                dt = datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                dt = None
        dts.append(dt)

    # sort by path order as already sorted upstream; just pick first/last non-None
    dt_start = next((x for x in dts if x is not None), None)
    dt_end   = next((x for x in reversed(dts) if x is not None), None)

    if dt_start and dt_end:
        same_day = (dt_start.date() == dt_end.date())
        start_tag = dt_start.strftime('%Y%m%dT%H%M%S')
        end_tag   = dt_end.strftime('%H%M%S') if same_day else dt_end.strftime('%Y%m%dT%H%M%S')
        time_part = f"{start_tag}-{end_tag}"
    elif dt_start:
        time_part = dt_start.strftime('%Y%m%dT%H%M%S')
    else:
        time_part = "unknown_time"

    n = len(paths)
    return f"{prefix}_{time_part}_n{n}_concat"


def load_rhd_file(filepath=None, merge_files=False, sort_files=True, rebuild_time=True, verbose=False):
    """
    Load Intan `.rhd` file(s). If multiple files are selected/provided and
    `merge_files=True`, concatenate them along the time axis.

    Parameters
    ----------
    filepath : str | list[str] | tuple[str] | None
        Path to a single .rhd file, or a sequence of paths. If None, opens a
        file dialog (multi-select when merge_files=True).
    merge_files : bool
        If True and multiple files are provided/selected, concatenate them.
    sort_files : bool
        Sort paths lexicographically (filenames often encode time).
    rebuild_time : bool
        If True, rebuild a strictly monotonic t_amplifier after concatenation.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Parsed signal & metadata dictionary. When concatenating, arrays are
        stitched along the sample axis and metadata keys are harmonized.
    """
    # --------- Resolve filepaths ---------
    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    if filepath is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        if merge_files:
            paths = filedialog.askopenfilenames(
                title="Select one or more .rhd files (will be concatenated)",
                initialdir="/",
                filetypes=(("RHD files", "*.rhd"), ("All files", "*.*"))
            )
            paths = list(paths)
        else:
            p = filedialog.askopenfilename(
                title="Select a .rhd File",
                initialdir="/",
                filetypes=(("RHD files", "*.rhd"), ("All files", "*.*"))
            )
            paths = [p] if p else []
        if not paths:
            print("No file selected, returning None.")
            return None
    else:
        paths = _ensure_list(filepath)

    if len(paths) > 1 and not merge_files:
        raise ValueError("Multiple files provided but merge_files=False. "
                         "Pass a single path or set merge_files=True.")

    if sort_files and len(paths) > 1:
        paths = sorted(paths)
        if verbose:
            print("Paths to merge:")
            for path in paths:
                print(" -", path)

    # --------- single-file reader ---------
    def _read_one(path):
        if verbose:
            print(f"[load_rhd] Reading {os.path.basename(path)}")
        tic = time.time()
        fid = open(path, 'rb')
        filesize = os.path.getsize(path)

        # Read header and determine data size
        header = read_header(fid)
        data_present, filesize, num_blocks, num_samples = (
            calculate_data_size(header, path, fid, verbose)
        )

        # Read blocks (if any), then sanity-check EOF
        data = []
        if data_present:
            data = read_all_data_blocks(header, num_samples, num_blocks, fid, verbose)
            check_end_of_file(filesize, fid)

        # Populate result with header info
        result = {}
        header_to_result(header, result)

        # Parse & scale data (if present), apply notch as needed, store into result
        if data_present:
            parse_data(header, data)
            apply_notch_filter(header, data)
            data_to_result(header, data, result)

        # Annotate filename/path
        result['file_name'] = os.path.basename(path)
        result['file_path'] = path

        # Sample rate (Hz)
        fs = None
        try:
            fs = float(result['frequency_parameters']['amplifier_sample_rate'])
        except Exception:
            try:
                fs = float(header.get('sample_rate', None))
            except Exception:
                fs = None
        result["fs"] = fs           # legacy convenience
        result["_fs_Hz"] = fs       # canonical key used by merge branch

        # Infer sample count from typical arrays
        _n_samp = None
        for key in ("t_amplifier", "board_adc_data", "amplifier_data"):
            arr = result.get(key, None)
            if isinstance(arr, np.ndarray):
                _n_samp = arr.shape[-1]
                break
        result["n_samples"] = _n_samp

        if verbose:
            dur = (_n_samp / fs) if (fs and _n_samp) else None
            print(f"  -> samples: {_n_samp}, fs: {fs}, dur: {dur:.2f}s | {time.time() - tic:0.2f}s")

        # Single-file export helpers (safe: no 'merged' or 'results' here)
        stem = os.path.splitext(result['file_name'])[0]
        result['export_basename'] = stem
        result['export_basepath'] = os.path.dirname(result['file_path'])

        # Channel name convenience list
        if 'amplifier_channels' in result and isinstance(result['amplifier_channels'], (list, tuple)):
            try:
                result['channel_names'] = [ch.get('native_channel_name') for ch in result['amplifier_channels']]
            except Exception:
                n = len(result['amplifier_channels'])
                result['channel_names'] = [f"A-{i+1:03d}" for i in range(n)]
        else:
            result['channel_names'] = []

        return result

    # --------- no-merge fast-path ---------
    print(f'Filepaths to load: {paths}')
    if len(paths) == 1:
        # # path still might not be a file path but directory
        # path = get_file_paths(paths[0], '.rhd', verbose=verbose)
        # print(f'Expanded to .rhd files: {path}')
        # if len(path) > 1 and merge_files:
        #     paths = sorted(path) if sort_files else path
        # elif len(path) == 1:
        #     return _read_one(path[0])
        # else:
        #     raise ValueError(f"No .rhd files found in {paths[0]}")

        return _read_one(paths[0])

    # --------- merge multiple files ---------
    results = [_read_one(p) for p in paths]

    # Check sampling-rate compatibility
    fs_set = {r.get("_fs_Hz") or r.get("fs") for r in results}
    fs_set = {f for f in fs_set if f is not None}
    if len(fs_set) > 1:
        raise ValueError(f"Sampling rates differ across files: {fs_set}")

    merged = {}
    first = results[0]
    for k, v in first.items():
        merged[k] = v

    # Helpers for sample counts and concatenation
    def _samples_of(r):
        t = r.get("t_amplifier", None)
        if isinstance(t, np.ndarray):
            return t.size
        a = r.get("amplifier_data", None)
        if isinstance(a, np.ndarray):
            return a.shape[-1]
        return r.get("n_samples", None)

    fs = first.get("_fs_Hz") or first.get("fs")

    # Offsets/durations per segment
    concat_offsets = []
    concat_durations = []
    offset = 0
    for r in results:
        n = _samples_of(r) or 0
        concat_offsets.append(offset)
        concat_durations.append((n / fs) if (fs and n) else np.nan)
        offset += n

    # Concatenate ndarray fields present in any result
    all_keys = set().union(*[set(r.keys()) for r in results])

    def _cat(key, arrays):
        arrays = [a for a in arrays if isinstance(a, np.ndarray)]
        if not arrays:
            return None
        ndim = arrays[0].ndim
        if ndim == 2:
            n_ch = arrays[0].shape[0]
            for a in arrays[1:]:
                if a.shape[0] != n_ch:
                    raise ValueError(f"Channel mismatch for '{key}': {a.shape[0]} vs {n_ch}")
            return np.concatenate(arrays, axis=1)
        elif ndim == 1:
            return np.concatenate(arrays, axis=0)
        else:
            lead = arrays[0].shape[:-1]
            for a in arrays[1:]:
                if a.shape[:-1] != lead:
                    raise ValueError(f"Shape mismatch for '{key}': {a.shape} vs {arrays[0].shape}")
            return np.concatenate(arrays, axis=-1)

    for key in all_keys:
        if key.startswith("_"):
            continue
        arrays = [r.get(key, None) for r in results]
        if any(isinstance(a, np.ndarray) for a in arrays):
            try:
                merged[key] = _cat(key, arrays)
            except Exception as e:
                if verbose:
                    print(f"[merge] Skipping concat for '{key}': {e}")
                merged[key] = first.get(key, None)

    # Merge names/paths, annotate, optionally rebuild time to be monotonic
    merged['file_name'] = ";".join([r.get('file_name', "") for r in results])
    merged['file_path'] = ";".join([r.get('file_path', "") for r in results])
    merged['_fs_Hz'] = fs
    merged['concat_segment_offsets_samples'] = np.array(concat_offsets, dtype=int)
    merged['concat_segment_durations_s'] = np.array(concat_durations, dtype=float)

    if rebuild_time and fs:
        n = _samples_of(merged) or 0
        merged["t_amplifier"] = np.arange(n, dtype=float) / fs

    if verbose:
        total_samp = _samples_of(merged) or 0
        total_dur = (total_samp / fs) if (fs and total_samp) else None
        print(f"[merge] concatenated {len(results)} files → samples={total_samp}, dur={total_dur:.2f}s")

    # Canonical export basename/path and channel names
    original_paths = [r.get('file_path') for r in results if r.get('file_path')]
    merged['export_basename'] = _build_export_basename(original_paths)
    merged['export_basepath'] = os.path.dirname(merged['file_path'].split(';')[0])

    if 'amplifier_channels' in results[0]:
        try:
            merged['channel_names'] = [ch.get('native_channel_name') for ch in results[0]['amplifier_channels']]
        except Exception:
            n = len(results[0]['amplifier_channels'])
            merged['channel_names'] = [f"A-{i+1:03d}" for i in range(n)]
    else:
        merged['channel_names'] = []

    return merged



def read_time_file(path):
    """
    Reads int32 timestamp values from a time.dat file.

    Parameters:
        path (str): Path to the time.dat file.

    Returns:
        np.ndarray: Array of timestamps in microseconds.
    """
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        time_data = np.fromfile(f, dtype=np.int32)
    return time_data


def read_amplifier_file(path, num_channels):
    """
    Load amplifier signal from a `.dat` file (One File Per Signal Type format).

    Parameters:
        path (str): Full path to `amplifier.dat`
        num_channels (int): Number of amplifier channels recorded

    Returns:
        np.ndarray: Amplifier signals (channels × samples) in µV.
    """
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)
    reshaped = data.reshape((-1, num_channels)).T
    return reshaped * 0.195  # Convert to microvolts


def read_auxiliary_file(path, num_channels, scale=0.0000374):
    """
    Reads auxiliary channel data (uint16) and applies scaling.
    """
    # First check if the file exists
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
    reshaped = data.reshape((-1, num_channels)).T
    return reshaped * scale


def read_adc_file(path, num_channels, scale=0.000050354):
    """
    Reads board ADC data (uint16) and applies default scaling.

    Parameters:
        path (str): Path to the board_adc.dat file.
        num_channels (int): Number of ADC channels recorded.
        scale (float): Scaling factor for ADC data.

    Returns:
        np.ndarray: Board ADC data (channels × samples) in Volts.
    """
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
    reshaped = data.reshape((-1, num_channels)).T
    return reshaped * scale


def load_dat_file(filepath=None):
    """
    Load dataset in 'One File Per Signal Type' format using external `.dat` files.

    Requires presence of an `info.rhd` file in the same directory for channel metadata.

    Parameters:
        filepath (str or None): Path to any `.dat` file in the dataset. Opens file dialog if None.

    Returns:
        dict: Parsed signal data and metadata.
    """
    if not filepath:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        filepath = filedialog.askdirectory()
        if not filepath:
            print("No folder selected, returning.")
            return None, False

        # TO-DO: support loading from a single file
        #filepath = filedialog.askopenfilename()
        #if not filepath:
        #    print("No file selected, returning.")
        #    return None, False
    # file_dir = os.path.dirname(filepath)

    # Attempt to read header information from an .rhd file if it exists
    if not os.path.exists(os.path.join(filepath, 'info.rhd')):
        print("No info.rhd file found in the directory.")
        return None, False
    file_name = os.path.join(filepath, 'info.rhd')

    # Load the header information
    header = load_rhd_file(file_name, verbose=False)

    # Now load the associated .dat files (One File Per Signal Type format)
    result = load_per_signal_files(filepath, header)

    # Add header keys to the result
    for key in header.keys():
        if key not in result:
            result[key] = header[key]

    return result


def load_per_signal_files(folder_path, header):
    """
    Load all .dat files in the specified folder, using the header information

    Parameters:
        folder_path (str): Path to the folder containing .dat files.
        header (dict): Header information from the .rhd file.

    Returns:
        dict: Dictionary containing all loaded data and metadata.
    """
    result = {}

    file_tasks = [
        ('t_amplifier', "time.dat", read_time_file, None),
        ('amplifier_data', "amplifier.dat", read_amplifier_file, len(header['amplifier_channels'])),
        ('aux_input_data', "auxiliary.dat", read_auxiliary_file, len(header['aux_input_channels'])),
    ]
    if 'board_acd_channels' in header:
        file_tasks.append(('board_adc_data', "board_adc.dat", read_adc_file,
                           len(header['board_adc_channels']))) if 'board_adc_channels' in header else None

    num_files = len(file_tasks)
    print("Reading .dat files...")
    print_step = 10
    percent_done = print_step

    # Loop through each task
    for i, (key, filename, read_function, channels) in enumerate(file_tasks):
        filepath = os.path.join(folder_path, filename)
        if channels is not None:
            result[key] = read_function(filepath, channels)
        else:
            result[key] = read_function(filepath)

        # Progress print
        percent_done = print_progress(i + 1, num_files, print_step, percent_done)

    # Add time vectors
    result['amplifier_channels'] = header['amplifier_channels']
    result['t_aux_input'] = result['t_amplifier'][::4]
    result['t_board_adc'] = result['t_amplifier']

    # Frequency parameters
    result['frequency_parameters'] = header['frequency_parameters']

    return result


def load_files_from_path(folder_path=None, concatenate=False):
    """
    Loads all .rhd files from a specified path or using a file dialog. Concatenates teh data if specified.

    Optionally concatenate the data from all files into a single result dictionary.

    Parameters:
        folder_path (str or None): The path to the folder containing the .rhd files.
        concatenate (bool): Boolean indicating if the data from all files should be concatenated.

    Returns:
        all_results: A list of 'result' dictionaries if concatenate is False, otherwise a single 'result' dictionary.
    """
    if folder_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        folder_path = filedialog.askdirectory()
        if not folder_path:
            print("No file selected, returning.")
            return None

    # Get the absolute paths of all files locates in the directory
    file_list = get_file_paths(folder_path)
    all_results = None
    for file in file_list:
        result = load_rhd_file(file, verbose=False)
        if not result:
            continue

        if concatenate:
            if all_results is None:
                all_results = result
            else:
                # Assuming all_results has the same fields, update the specific ones
                keys = ['t_aux_input', 'aux_input_data', 't_amplifier', 'amplifier_data', 't_board_adc',
                        'board_adc_data', ]
                for key in keys:
                    original_data = all_results[key]
                    new_data = result[key]
                    # If there is more than oen column, concatenate along axis=1, otherwise axis=0
                    if len(original_data.shape) > 1:
                        all_results[key] = np.concatenate((original_data, new_data), axis=1)
                    else:
                        all_results[key] = np.concatenate((original_data, new_data), axis=0)
        else:
            if all_results is None:
                all_results = [result]
            else:
                all_results.append(result)

    return all_results

def apply_notch_filter(header, data, verbose=True):
    """Checks header to determine if notch filter should be applied, and if so,
    apply notch filter to all signals in data['amplifier_data'].

    Parameters:
        header (dict): The header information of the data file.
        data (dict): The raw data to be parsed.
        verbose (bool): If True, print progress messages. Default is True.
    """
    # If data was not recorded with notch filter turned on, return without
    # applying notch filter. Similarly, if data was recorded from Intan RHX
    # software version 3.0 or later, any active notch filter was already
    # applied to the saved data, so it should not be re-applied.
    if (header['notch_filter_frequency'] == 0
            or header['version']['major'] >= 3):
        return

    # Apply notch filter individually to each channel in order
    print('Applying notch filter...')
    print_step = 10
    percent_done = print_step
    for i in range(header['num_amplifier_channels']):
        data['amplifier_data'][i, :] = notch_filter(
            data['amplifier_data'][i, :],
            header['sample_rate'],
            header['notch_filter_frequency'],
            10)

        if verbose:
            percent_done = print_progress(i + 1, header['num_amplifier_channels'],
                                          print_step, percent_done)



def parse_data(header, data):
    """Parses raw data into user readable and interactable forms (for example,
    extracting raw digital data to separate channels and scaling data to units
    like microVolts, degrees Celsius, or seconds.)

    Parameters:
        header (dict): The header information of the data file.
        data (dict): The raw data to be parsed.
    """
    print('Parsing data...')
    extract_digital_data(header, data)
    scale_analog_data(header, data)
    scale_timestamps(header, data)


def scale_timestamps(header, data):
    """Verifies no timestamps are missing, and scales timestamps to seconds.

    Parameters:
        header (dict): The header information of the data file.
        data (dict): The raw data to be parsed.
    """
    # Check for gaps in timestamps.
    num_gaps = np.sum(np.not_equal(
        data['t_amplifier'][1:] - data['t_amplifier'][:-1], 1))
    if num_gaps == 0:
        print('No missing timestamps in data.')
    else:
        print('Warning: {0} gaps in timestamp data found.  '
              'Time scale will not be uniform!'
              .format(num_gaps))

    # Scale time steps (units = seconds).
    data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
    data['t_aux_input'] = data['t_amplifier'][range(
        0, len(data['t_amplifier']), 4)]
    data['t_supply_voltage'] = data['t_amplifier'][range(
        0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
    data['t_board_adc'] = data['t_amplifier']
    data['t_dig'] = data['t_amplifier']
    data['t_temp_sensor'] = data['t_supply_voltage']


def scale_analog_data(header, data):
    """Scales all analog data signal types (amplifier data, aux input data,
    supply voltage data, board ADC data, and temp sensor data) to suitable
    units (microVolts, Volts, deg C).

    Parameters:
        header (dict): The header information of the data file.
        data (dict): The raw data to be parsed.
    """
    # Scale amplifier data (units = microVolts).
    data['amplifier_data'] = np.multiply(
        0.195, (data['amplifier_data'].astype(np.int32) - 32768))

    # Scale aux input data (units = Volts).
    data['aux_input_data'] = np.multiply(
        37.4e-6, data['aux_input_data'])

    # Scale supply voltage data (units = Volts).
    data['supply_voltage_data'] = np.multiply(
        74.8e-6, data['supply_voltage_data'])

    # Scale board ADC data (units = Volts).
    if header['eval_board_mode'] == 1:
        data['board_adc_data'] = np.multiply(
            152.59e-6, (data['board_adc_data'].astype(np.int32) - 32768))
    elif header['eval_board_mode'] == 13:
        data['board_adc_data'] = np.multiply(
            312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768))
    else:
        data['board_adc_data'] = np.multiply(
            50.354e-6, data['board_adc_data'])

    # Scale temp sensor data (units = deg C).
    data['temp_sensor_data'] = np.multiply(
        0.01, data['temp_sensor_data'])


def extract_digital_data(header, data):
    """Extracts digital data from raw (a single 16-bit vector where each bit
    represents a separate digital input channel) to a more user-friendly 16-row
    list where each row represents a separate digital input channel. Applies to
    digital input and digital output data.

    Parameters:
        header (dict): The header information of the data file.
        data (dict): The raw data to be parsed.
    """
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_in_raw'],
                (1 << header['board_dig_in_channels'][i]['native_order'])
            ),
            0)

    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_out_raw'],
                (1 << header['board_dig_out_channels'][i]['native_order'])
            ),
            0)

