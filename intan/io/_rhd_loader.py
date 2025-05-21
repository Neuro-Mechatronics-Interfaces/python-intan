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
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
from intan.io._header_parsing import read_header, header_to_result, data_to_result
from intan.io._metadata_utils import calculate_data_size
from intan.io._block_parser import read_all_data_blocks
from intan.io._file_utils import check_end_of_file, print_progress, get_file_paths
#from intan.processing._filters import notch_filter


def load_rhd_file(filepath=None, verbose=True):
    """
    Load a full `.rhd` file recorded by Intan Technologies hardware.

    Uses the embedded header to decode all available signals, applies any
    recording-time filtering settings, and returns a structured result.

    Parameters:
        filepath (str or None): Path to the .rhd file. If None, opens file dialog.
        verbose (bool): Whether to print file summary and timing.

    Returns:
        dict: Parsed signal and metadata dictionary.
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
        root_dir = filedialog.askdirectory()
        if not root_dir:
            print("No folder selected, returning.")
            return None, False

        # TO-DO: support loading from a single file
        #filepath = filedialog.askopenfilename()
        #if not filepath:
        #    print("No file selected, returning.")
        #    return None, False
    # file_dir = os.path.dirname(filepath)

    # Attempt to read header information from an .rhd file if it exists
    if not os.path.exists(os.path.join(root_dir, 'info.rhd')):
        print("No info.rhd file found in the directory.")
        return None, False
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

