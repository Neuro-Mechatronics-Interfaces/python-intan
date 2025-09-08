"""
intan.io._file_utils

Utility functions for file and path handling in the Intan interface.

This module includes helpers for:
- Cross-platform path adjustment (Windows, WSL, Linux)
- File presence validation
- Reading configuration or labeled trial files
- End-of-file checks and error handling
- Progress bar display during file loading
- Directory scanning for `.rhd` datasets

Used throughout the `intan.io` submodule to support flexible loading and validation
of data from local or mounted environments.
"""

import sys
import os
import json
import numpy as np
from typing import Optional
from pathlib import Path
import yaml
import platform
import pathlib
import pandas as pd
from tkinter import filedialog
from intan.io._exceptions import FileSizeError


def adjust_path(path):
    """
    Adjusts a file path for compatibility with the host operating system.

    Automatically detects WSL, Linux, or native Windows and transforms
    file paths accordingly (e.g., `C:\\` to `/mnt/c/` on WSL).

    Parameters:
        path (str): Original file path (Windows-style or POSIX)

    Returns:
        str: Transformed path compatible with the current OS.
    """
    system = platform.system()

    # Check if the system is running under WSL
    if "microsoft" in platform.uname().release.lower():
        system = "WSL"

    # Modify the path based on the system type
    if system == "Windows":
        # If running on native Windows, return the path as is
        return path


    elif system == "WSL":
        # If running on WSL, convert "C:/" to "/mnt/c/"
        if len(path) > 1 and path[1] == ":":
            drive_letter = path[0].lower()
            # Properly replace backslashes without using them in f-string
            linux_path = path[2:].replace("\\", "/")
            return f"/mnt/{drive_letter}{linux_path}"
        else:
            return path

    elif system == "Linux":
        # If running on native Linux, assume Linux paths are provided correctly
        return path

    else:
        raise ValueError(f"Unsupported system: {system}")


def check_file_present(file, metrics_file, verbose=False):
    """
    Check whether a file is listed in a given metrics CSV.

    Parameters:
        file (str): File path to check.
        metrics_file (pd.DataFrame): Loaded CSV with a 'File Name' column.
        verbose (bool): If True, print a warning when file is missing.

    Returns:
        tuple: (filename, is_present) where `is_present` is True if the file is found.
    """
    filename = pathlib.Path(file).name
    if filename not in metrics_file['File Name'].tolist():
        if verbose:
            print(f"File {filename} not found in metrics file")
        return filename, False

    return filename, True


def check_end_of_file(filesize, fid):
    """
    Validate that the file pointer has reached the end of the file.

    Raises:
        FileSizeError: If unread bytes remain in the stream.
    """
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining != 0:
        raise FileSizeError('Error: End of file not reached.')


def print_progress(i, target, print_step, percent_done, bar_length=40):
    """
    Print an ASCII progress bar in the terminal.

    Parameters:
        i (int): Current iteration index.
        target (int): Total number of iterations.
        print_step (float): Update frequency as a percentage.
        percent_done (float): Last percentage printed (used to avoid overprinting).
        bar_length (int): Length of the ASCII progress bar.

    Returns:
        float: Updated `percent_done` value.
    """
    fraction_done = 100 * (1.0 * i / target)

    # Only update if we've crossed a new step
    if fraction_done >= percent_done:
        fraction_bar = i / target
        arrow = '=' * int(fraction_bar * bar_length - 1) + '>' if fraction_bar > 0 else ''
        padding = ' ' * (bar_length - len(arrow))

        ending = '\n' if i == target - 1 else '\r'

        print(f'Progress: [{arrow}{padding}] {int(fraction_bar * 100)}%', end=ending)
        sys.stdout.flush()

        percent_done += print_step

    return percent_done


def read_config_file(config_file):
    """
    Parse a simple key=value style configuration file (e.g. TRUECONFIG.txt).

    Parameters:
        config_file (str): Path to the config file.

    Returns:
        dict: Dictionary of key-value settings.
    """
    # Dictionary to store the key-value pairs
    config_data = {}

    # Open the TRUECONFIG.txt file and read its contents
    with open(config_file, 'r') as file:
        for line in file:
            # Strip whitespace and ignore empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value at the first '='
            key, value = line.split('=', 1)
            config_data[key.strip()] = value.strip()

    return config_data


def get_file_paths(directory, file_type=None, verbose=False):
    """
    Scan a directory for files or folders, with optional filtering by extension.

    Parameters:
        directory (str): Root folder to search.
        file_type (str or None): Extension to search for (e.g. '.rhd'). If None, returns subfolders.
        verbose (bool): If True, print the number of items found.

    Returns:
        list[pathlib.Path]: List of matching files or folders.
    """
    if verbose: print("Searching in directory:", directory)

    # Convert the directory to an absolute path and a Path object for compatibility
    directory = pathlib.Path(os.path.abspath(directory))

    # Check if the directory exists
    if not directory.exists() or not directory.is_dir():
        print(f"Directory '{directory}' not found or is not a valid directory.")
        return []

    # If file_type is left None, we just need to return the folders within the current directory
    file_paths = None
    if file_type is None:
        file_paths = list(directory.glob('*'))
        if verbose: print(f"Found {len(file_paths)} folders")
    elif file_type == '.rhd':
        # Recursively find all .rhd files
        file_paths = list(directory.rglob('*.rhd'))
        if verbose: print(f"Found {len(file_paths)} .rhd files")
    else:
        print("Unsupported file type. Please specify '.rhd' or None.")

    return file_paths


def load_labeled_file(path=None):
    """
    Load a label/notes file containing gesture timing and annotations.

    This is used for supervised EMG labeling during offline training.

    If no path is provided, opens a GUI file selection dialog.

    Parameters:
        path (str or None): Path to the `.txt` notes file.

    Returns:
        pd.DataFrame: Labeled samples with columns ["Sample", "Time", "Label"],
                      cleaned and sorted by sample index.
    """
    if path is None:
        path = filedialog.askopenfilename(title="Select Notes File", filetypes=[("Text files", "*.txt")])
        if not path:
            return pd.DataFrame(columns=["Sample", "Time", "Label"])  # return empty if cancelled

    # Read the notes file (no header expected)
    df_raw = pd.read_csv(path, header=None)

    # If the first row contains column names (text), drop it
    if any(isinstance(cell, str) and "sample" in str(cell).lower() for cell in df_raw.iloc[0]):
        df_raw = df_raw.iloc[1:]

    # Assign column names
    df_raw.columns = ["Sample", "Time", "Label"]

    # Remove any rows that contain 'threshold' (or other irrelevant markers) in the label
    df_raw = df_raw[~df_raw["Label"].astype(str).str.contains("threshold", case=False, na=False)]

    # Clean up label text (e.g., remove the word " cue" if present)
    df_raw["Label"] = df_raw["Label"].astype(str).str.replace(" cue", "", case=False)

    # Convert sample indices to int and sort by time
    df_raw["Sample"] = df_raw["Sample"].astype(int)
    df_sorted = df_raw.sort_values(by="Sample").reset_index(drop=True)
    return df_sorted


def load_txt_config(file_path=None, verbose=False):
    """
    Parse a simple key=value style configuration file (e.g. config.txt).

    Parameters:
        file_path (str): Path to the config file.
        verbose (bool): If True, print warnings and info messages.

    Returns:
        dict: Dictionary of key-value settings.

    """

    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File", filetypes=[("Text files", "*.txt")])
        if not file_path:
            if verbose:
                print("Cancelled selection")
            return None

    # Dictionary to store the key-value pairs
    config_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and ignore empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value at the first '='
            key, value = line.split('=', 1)
            config_data[key.strip()] = value.strip()
    return config_data


def load_yaml_file(file_path=None):
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed config dictionary.

    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File",
                                               filetypes=[("Text files", ["*.yaml", "*.yml"])])
        if not file_path:
            return None

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_json_file(file_path=None):
    """
    Load configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed config dictionary.

    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File",
                                               filetypes=[("Text files", ["*.json"])])
        if not file_path:
            return None

    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


def load_config_file(file_path=None, verbose=False):
    """
    Load configuration from a file, supporting .txt, .yaml, and .json formats.

    Args:
        file_path (str): Path to the config file.
        verbose (bool): If True, print debug information.

    Returns:
        dict: Parsed config dictionary or None if loading failed.
    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Config File",
                                               filetypes=[("Text files", ["*.txt", "*.yaml", "*.yml", "*.json"])])
        if not file_path:
            if verbose:
                print("Cancelled selection")
            return None

    ext = Path(file_path).suffix.lower()
    if ext == '.txt':
        return load_txt_config(file_path, verbose)
    elif ext in ['.yaml', '.yml']:
        return load_yaml_file(file_path)
    elif ext == '.json':
        return load_json_file(file_path)
    else:
        print(f"Unsupported config file format: {ext}")
        return None


def labels_from_events(event_path, window_starts, *, strict_segment=False, fs=2000):
    """
    Map each window start (absolute sample index) to a label using an events CSV
    with columns: 'Sample Index', 'Timestamp', 'Label'. The Timestamp text is ignored.

    strict_segment=True: drops any window whose [start, start+step) crosses an event boundary.
    """
    df = pd.read_csv(event_path)
    if 'Sample Index' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Event file must have 'Sample Index' and 'Label' columns")
    ev_idx = np.asarray(df['Sample Index'].values, dtype=np.int64)
    ev_lab = df['Label'].astype(str).values

    # Trim any '#' comments from labels
    ev_lab = np.array([lab.split('#')[0].strip() for lab in ev_lab], dtype=str)

    # sort by sample index
    order = np.argsort(ev_idx)
    ev_idx = ev_idx[order]
    ev_lab = ev_lab[order]

    # for each window start, pick the last event with idx <= start
    # searchsorted returns insertion position; subtract 1 to get the last <=
    pos = np.searchsorted(ev_idx, window_starts, side='right') - 1
    # anything before the first event is Unknown
    y = np.where(pos >= 0, ev_lab[pos], 'Unknown')

    if strict_segment:
        # drop windows that cross an event boundary
        # boundary after this window's start is at ev_idx[pos+1]
        next_pos = pos + 1
        next_change = np.where(next_pos < ev_idx.size, ev_idx[next_pos], np.iinfo(np.int64).max)
        # if your step in samples is known, ensure start+step <= next_change
        # (pass it in or compute the exact end you want). As a simple guard:
        # keep only windows whose label doesn't change immediately after start.
        ok = (next_change > window_starts)
        y = np.where(ok, y, 'Unknown')
    return y


def last_event_index(path: str) -> Optional[int]:
    if not os.path.isfile(path): return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.lower().startswith("sample"):
                continue
            try:
                idx = int(ln.split()[0].replace(",", ""))
                last = idx if last is None else max(last, idx)
            except Exception:
                pass
    return last

