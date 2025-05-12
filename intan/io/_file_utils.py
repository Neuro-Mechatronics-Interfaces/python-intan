import sys
import os
import platform
import pathlib
import pandas as pd
from tkinter import filedialog
from intan.io._exceptions import FileSizeError

def adjust_path(path):
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
    Checks if the file is present in the metrics file.

    Args:
        file: The file to check.
        metrics_file: The metrics file to search.
        verbose:  (Optional) Boolean indicating if the function should print messages.

    Returns:
        filename: The name of the file.
        is_present: Boolean indicating if the file is present in the metrics file.
    """
    filename = pathlib.Path(file).name
    if filename not in metrics_file['File Name'].tolist():
        if verbose:
            print(f"File {filename} not found in metrics file")
        return filename, False

    return filename, True

def check_end_of_file(filesize, fid):
    """Checks that the end of the file was reached at the expected position.
    If not, raise FileSizeError.
    """
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining != 0:
        raise FileSizeError('Error: End of file not reached.')


def print_progress(i, target, print_step, percent_done, bar_length=40):
    """Prints an updating progress bar in the terminal while respecting print_step and percent_done."""
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
    Returns a list of full paths for files ending with .rhd in the given directory and its subdirectories.

    Parameters:
    ------------
    directory:    (str) The parent directory to search within.
    file_type:    (str) The file extension to search for (default: '.rhd').
    verbose:      (bool) Whether to print the number of files found.

    Returns:
        rhd_files: List of full paths to either folders or files
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
    Load a notes text file containing trial timing and labels.
    If `path` is None, opens a file dialog for the user to select a .txt file.
    Returns a DataFrame with columns ["Sample", "Time", "Label"], sorted by Sample index.
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
