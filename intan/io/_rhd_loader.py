import os
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
from intan.io._header_parsing import read_header, header_to_result, data_to_result
from intan.io._metadata_utils import calculate_data_size
from intan.io._block_parser import read_all_data_blocks
from intan.io._file_utils import check_end_of_file, print_progress, get_file_paths
from intan.io._data_processing import parse_data, apply_notch_filter

def load_rhd_file(filepath=None, verbose=True):
    """Loads .rhd file with provided filename, returning 'result' dict and
    'data_present' Boolean.
    """
    if not filepath:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        filepath = filedialog.askopenfilename()
        if not filepath:
            print("No file selected, returning.")
            return None, False

    # Start timing
    tic = time.time()

    # Open file
    fid = open(filepath, 'rb')
    filesize = os.path.getsize(filepath)

    # Read file header
    header = read_header(fid)

    # Calculate how much data is present and summarize to console.
    data_present, filesize, num_blocks, num_samples = (
        calculate_data_size(header, filepath, fid, verbose))

    # If .rhd file contains data, read all present data blocks into 'data'
    # dict, and verify the amount of data read.
    if data_present:
        data = read_all_data_blocks(header, num_samples, num_blocks, fid, verbose)
        check_end_of_file(filesize, fid)

    # Save information in 'header' to 'result' dict.
    result = {}
    header_to_result(header, result)

    # If .rhd file contains data, parse data into readable forms and, if
    # necessary, apply the same notch filter that was active during recording.
    if data_present:
        parse_data(header, data)
        apply_notch_filter(header, data)

        # Save recorded data in 'data' to 'result' dict.
        data_to_result(header, data, result)

    # Otherwise (.rhd file is just a header for One File Per Signal Type or
    # One File Per Channel data formats, in which actual data is saved in
    # separate .dat files), just return data as an empty list.
    else:
        data = []

    # Save filename to result
    result['file_name'] = os.path.basename(filepath)
    result['file_path'] = filepath

    # Report how long read took.
    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))

    # Return 'result' dict.
    return result

def read_time_file(path):
    """Reads int32 timestamp values from a time.dat file."""
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        time_data = np.fromfile(f, dtype=np.int32)
    return time_data

def read_amplifier_file(path, num_channels):
    """Reads amplifier data from a .dat file (int16 format) and reshapes it."""
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)
    reshaped = data.reshape((-1, num_channels)).T
    return reshaped * 0.195  # Convert to microvolts

def read_auxiliary_file(path, num_channels, scale=0.0000374):
    """Reads auxiliary channel data (uint16) and applies scaling."""
    # First check if the file exists
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
    reshaped = data.reshape((-1, num_channels)).T
    return reshaped * scale

def read_adc_file(path, num_channels, scale=0.000050354):
    """Reads board ADC data (uint16) and applies default scaling."""
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
    reshaped = data.reshape((-1, num_channels)).T
    return reshaped * scale

def load_dat_file(root_dir=None):

    if root_dir is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        filepath = filedialog.askdirectory()
        if not filepath:
            print("No path selected, returning.")
            return None, False

    # Read the header information from the .rhd file
    file_name = os.path.join(root_dir, 'info.rhd')

    # Load the header information
    header = load_rhd_file(file_name, verbose=False)

    # Now load the associated .dat files (One File Per Signal Type format)
    result = load_per_signal_files(root_dir, header)

    # Add header keys to the result
    for key in header.keys():
        if key not in result:
            result[key] = header[key]

    return result

def load_per_signal_files(folder_path, header):

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
        percent_done = print_progress(i+1, num_files, print_step, percent_done)

    # Add time vectors
    result['amplifier_channels'] = header['amplifier_channels']
    result['t_aux_input'] = result['t_amplifier'][::4]
    result['t_board_adc'] = result['t_amplifier']

    # Frequency parameters
    result['frequency_parameters'] = header['frequency_parameters']

    return result

def load_files_from_path(folder_path=None, concatenate=False):
    """ Loads all .rhd files from a specified path or using a file dialog. Concatenates teh data if specified.

    Args:
        folder_path: The path to the folder containing the .rhd files.
        concatenate: Boolean indicating if the data from all files should be concatenated.

    Returns:
        all_results: A list of 'result' dictionaries if concatenate is False, otherwise a single 'result' dictionary.
        success: Boolean indicating if the files were loaded successfully.
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
                keys = ['t_aux_input', 'aux_input_data', 't_amplifier', 'amplifier_data', 't_board_adc','board_adc_data', ]
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
