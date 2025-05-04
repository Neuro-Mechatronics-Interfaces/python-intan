import sys
import os
import platform
from pathlib import Path
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
    filename = Path(file).name
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

def get_rhd_file_paths(directory, verbose=False):
    """
    Returns a list of full paths for files ending with .rhd in the given directory and its subdirectories.

    Args:
        directory: The directory to search for .rhd files.

    Returns:
        rhd_files: List of full paths to .rhd files.
    """
    if verbose:
        print("Searching in directory:", directory)

    # Convert the directory to an absolute path and a Path object for compatibility
    directory = Path(os.path.abspath(directory))

    # Check if the directory exists
    if not directory.exists() or not directory.is_dir():
        print(f"Directory '{directory}' not found or is not a valid directory.")
        return []

    # Recursively find all .rhd files
    file_paths = list(directory.rglob('*.rhd'))
    if verbose:
        print(f"Found {len(file_paths)} .rhd files")
    return file_paths

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
