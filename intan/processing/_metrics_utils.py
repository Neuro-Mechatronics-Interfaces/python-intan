import os
import pathlib
import pandas as pd

def read_config_file(config_file):
    # Dictionary to store the key-value pairs
    config_data = {}

    config_filepath = pathlib.Path(config_file)

    # Open the TRUECONFIG.txt file and read its contents
    with open(config_filepath, 'r') as file:
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

def load_metrics_data(metrics_filepath, verbose=True):
    """ Loads the metrics data from the specified file path and returns the data along with the gesture mapping.

    Args:
        metrics_filepath (str): The path to the metrics data file.
        verbose    (bool): Whether to print the loaded data and gesture mapping.
    """
    if not os.path.isfile(metrics_filepath):
        print(f"Metrics file not found: {metrics_filepath}. Please correct file path or generate the metrics file.")
        return None
    metrics_data = pd.read_csv(metrics_filepath)
    if verbose:
        print(f"Loaded metrics data from {metrics_filepath}: unique labels {metrics_data['Gesture'].unique()}")
        print(metrics_data)

    # Generate gesture mapping
    gestures = metrics_data['Gesture'].unique()
    gesture_map = {gesture: i for i, gesture in enumerate(gestures)}
    if verbose:
        print(f"Gesture mapping: {gesture_map}")

    return metrics_data, gesture_map

