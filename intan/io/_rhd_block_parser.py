
"""
intan.io._block_parser

This module provides low-level parsing functions for reading binary `.rhd` data blocks
produced by Intan Technologies hardware. It handles analog/digital signal unpacking,
timestamp alignment, and memory preallocation for signal extraction.

Functions include:
- Reading amplifier and auxiliary signals
- Extracting timestamps
- Assembling full data dictionaries

Intended for internal use by the Intan RHX Python interface.
"""
import numpy as np
import struct
from intan.io._file_utils import print_progress

FRAMES_PER_BLOCK = 128
MAGIC_NUMBER = 0x2ef07a08

SAMPLE_SCALE_FACTOR = 0.195  # Scale factor for amplifier data
ANALOG_SCALE_FACTOR = 312.5e-6  # Scale factor for analog data


# Byte Parsing Functions
def read_uint32(array, arrayIndex):
    """Read a 4-byte unsigned integer from the data."""
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    return variable, arrayIndex + 4


def read_int32(array, arrayIndex):
    """Read a 4-byte signed integer from the data."""
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    return variable, arrayIndex + 4


def read_uint16(array, arrayIndex):
    """Read a 2-byte unsigned integer from the data."""
    variableBytes = array[arrayIndex: arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    return variable, arrayIndex + 2


def get_timestamp_signed(header):
    """
    Determine if timestamps are stored as signed integers.

    This depends on the version of the Intan file format.
    Intan software version 1.2 and later uses signed timestamps.

    Parameters:
        header (dict): Parsed header dictionary containing version info.
    Returns:
        bool: True if timestamps are signed; False if unsigned.
    """
    # All Intan software v1.2 and later saves timestamps as signed
    if header['version']['major'] > 1:
        return True

    if header['version']['major'] == 1 and header['version']['minor'] >= 2:
        return True

    # Intan software before v1.2 saves timestamps as unsigned
    return False


def read_one_data_block(data, header, indices, fid):
    """Reads one 60 or 128 sample data block from fid into data,
    at the location indicated by indices

    Parameters:
        data (dict): Dictionary to store the read data.
        header (dict): Header information from the file.
        indices (dict): Indices for each signal type in the data dictionary.
        fid: File object to read from.
    """

    samples_per_block = header['num_samples_per_data_block']

    # In version 1.2, we moved from saving timestamps as unsigned
    # integers to signed integers to accommodate negative (adjusted)
    # timestamps for pretrigger data
    read_timestamps(fid,
                    data,
                    indices,
                    samples_per_block,
                    get_timestamp_signed(header))

    read_analog_signals(fid,
                        data,
                        indices,
                        samples_per_block,
                        header)

    read_digital_signals(fid,
                         data,
                         indices,
                         samples_per_block,
                         header)


def read_timestamps(fid, data, indices, num_samples, timestamp_signed):
    """Reads timestamps from binary file as a NumPy array, indexing them
    into 'data'.

    Parameters:
        fid: File object to read from.
        data (dict): Dictionary to store the read data.
        indices (dict): Indices for each signal type in the data dictionary.
        num_samples (int): Number of samples to read.
        timestamp_signed (bool): Flag indicating if timestamps are signed.
    """
    start = indices['amplifier']
    end = start + num_samples
    format_sign = 'i' if timestamp_signed else 'I'
    format_expression = '<' + format_sign * num_samples
    read_length = 4 * num_samples
    data['t_amplifier'][start:end] = np.array(struct.unpack(
        format_expression, fid.read(read_length)))


def read_analog_signals(fid, data, indices, samples_per_block, header):
    """Reads all analog signal types present in RHD files: amplifier_data,
    aux_input_data, supply_voltage_data, temp_sensor_data, and board_adc_data,
    into 'data' dict.

    Parameters:
        fid: File object to read from.
        data (dict): Dictionary to store the read data.
        indices (dict): Indices for each signal type in the data dictionary.
        samples_per_block (int): Number of samples per block.
        header (dict): Header information from the file.
    """

    read_analog_signal_type(fid,
                            data['amplifier_data'],
                            indices['amplifier'],
                            samples_per_block,
                            header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['aux_input_data'],
                            indices['aux_input'],
                            int(samples_per_block / 4),
                            header['num_aux_input_channels'])

    read_analog_signal_type(fid,
                            data['supply_voltage_data'],
                            indices['supply_voltage'],
                            1,
                            header['num_supply_voltage_channels'])

    read_analog_signal_type(fid,
                            data['temp_sensor_data'],
                            indices['supply_voltage'],
                            1,
                            header['num_temp_sensor_channels'])

    read_analog_signal_type(fid,
                            data['board_adc_data'],
                            indices['board_adc'],
                            samples_per_block,
                            header['num_board_adc_channels'])


def read_digital_signals(fid, data, indices, samples_per_block, header):
    """Reads all digital signal types present in RHD files: board_dig_in_raw
    and board_dig_out_raw, into 'data' dict.

    Parameters:
        fid: File object to read from.
        data (dict): Dictionary to store the read data.
        indices (dict): Indices for each signal type in the data dictionary.
        samples_per_block (int): Number of samples per block.
        header (dict): Header information from the file.
    """

    read_digital_signal_type(fid,
                             data['board_dig_in_raw'],
                             indices['board_dig_in'],
                             samples_per_block,
                             header['num_board_dig_in_channels'])

    read_digital_signal_type(fid,
                             data['board_dig_out_raw'],
                             indices['board_dig_out'],
                             samples_per_block,
                             header['num_board_dig_out_channels'])


def read_analog_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be an analog signal type within 'data', for example
    data['amplifier_data'] or data['aux_input_data']. Each sample is assumed
    to be of dtype 'uint16'.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    tmp = np.fromfile(fid, dtype='uint16', count=num_samples * num_channels)
    dest[range(num_channels), start:end] = (
        tmp.reshape(num_channels, num_samples))


def read_digital_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be a digital signal type within 'data', either
    data['board_dig_in_raw'] or data['board_dig_out_raw'].

    Each sample is assumed to be of dtype 'uint16', and the data is
    unpacked into 'dest' as a 2D array of shape (num_channels, num_samples).

    Parameters:
        fid: File object to read from.
        dest (numpy.ndarray): Destination array to store the read data.
        start (int): Starting index for writing data.
        num_samples (int): Number of samples to read.
        num_channels (int): Number of channels to read.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    dest[start:end] = np.array(struct.unpack(
        '<' + 'H' * num_samples, fid.read(2 * num_samples)))


def read_all_data_blocks(header, num_samples, num_blocks, fid, verbose=True):
    """Reads all data blocks present in file, allocating memory for and
    returning 'data' dict containing all data.

    Parameters:
        header (dict): Header information from the file.
        num_samples (dict): Number of samples for each signal type.
        num_blocks (int): Number of blocks to read.
        fid: File object to read from.
        verbose (bool): Flag for verbose output.

    Returns:
        data (dict): Dictionary containing all read data.
    """
    data, indices = initialize_memory(header, num_samples)
    print("Reading data from file...")
    print_step = 10
    percent_done = print_step
    for i in range(num_blocks):
        read_one_data_block(data, header, indices, fid)
        advance_indices(indices, header['num_samples_per_data_block'])
        if verbose:
            percent_done = print_progress(i + 1, num_blocks, print_step, percent_done)
    return data


def initialize_memory(header, num_samples):
    """Pre-allocates NumPy arrays for each signal type that will be filled
    during this read, and initializes unique indices for data access to each
    signal type.

    Parameters:
        header (dict): Header information from the file.
        num_samples (dict): Number of samples for each signal type.

    Returns:
        data (dict): Dictionary with pre-allocated arrays for each signal type.
        indices (dict): Dictionary with indices for each signal type.

    """
    print('\nAllocating memory for data...')
    data = {}

    # Create zero array for amplifier timestamps.
    t_dtype = np.int_ if get_timestamp_signed(header) else np.uint
    data['t_amplifier'] = np.zeros(num_samples['amplifier'], t_dtype)

    # Create zero array for amplifier data.
    data['amplifier_data'] = np.zeros(
        [header['num_amplifier_channels'], num_samples['amplifier']],
        dtype=np.uint)

    # Create zero array for aux input data.
    data['aux_input_data'] = np.zeros(
        [header['num_aux_input_channels'], num_samples['aux_input']],
        dtype=np.uint)

    # Create zero array for supply voltage data.
    data['supply_voltage_data'] = np.zeros(
        [header['num_supply_voltage_channels'], num_samples['supply_voltage']],
        dtype=np.uint)

    # Create zero array for temp sensor data.
    data['temp_sensor_data'] = np.zeros(
        [header['num_temp_sensor_channels'], num_samples['supply_voltage']],
        dtype=np.uint)

    # Create zero array for board ADC data.
    data['board_adc_data'] = np.zeros(
        [header['num_board_adc_channels'], num_samples['board_adc']],
        dtype=np.uint)

    # By default, this script interprets digital events (digital inputs
    # and outputs) as booleans. if unsigned int values are preferred
    # (0 for False, 1 for True), replace the 'dtype=np.bool_' argument
    # with 'dtype=np.uint' as shown.
    # The commented lines below illustrate this for digital input data;
    # the same can be done for digital out.

    # data['board_dig_in_data'] = np.zeros(
    #     [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
    #     dtype=np.uint)
    # Create 16-row zero array for digital in data, and 1-row zero array for
    # raw digital in data (each bit of 16-bit entry represents a different
    # digital input.)
    data['board_dig_in_data'] = np.zeros(
        [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
        dtype=np.bool_)
    data['board_dig_in_raw'] = np.zeros(
        num_samples['board_dig_in'],
        dtype=np.uint)

    # Create 16-row zero array for digital out data, and 1-row zero array for
    # raw digital out data (each bit of 16-bit entry represents a different
    # digital output.)
    data['board_dig_out_data'] = np.zeros(
        [header['num_board_dig_out_channels'], num_samples['board_dig_out']],
        dtype=np.bool_)
    data['board_dig_out_raw'] = np.zeros(
        num_samples['board_dig_out'],
        dtype=np.uint)

    # Create dict containing each signal type's indices, and set all to zero.
    indices = {}
    indices['amplifier'] = 0
    indices['aux_input'] = 0
    indices['supply_voltage'] = 0
    indices['board_adc'] = 0
    indices['board_dig_in'] = 0
    indices['board_dig_out'] = 0

    return data, indices


def advance_indices(indices, samples_per_block):
    """Advances indices used for data access by suitable values per data block.

    Parameters:
        indices (dict): Dictionary with indices for each signal type.
        samples_per_block (int): Number of samples per block.
    """
    # Signal types sampled at the sample rate:
    # Index should be incremented by samples_per_block every data block.
    indices['amplifier'] += samples_per_block
    indices['board_adc'] += samples_per_block
    indices['board_dig_in'] += samples_per_block
    indices['board_dig_out'] += samples_per_block

    # Signal types sampled at 1/4 the sample rate:
    # Index should be incremented by samples_per_block / 4 every data block.
    indices['aux_input'] += int(samples_per_block / 4)

    # Signal types sampled once per data block:
    # Index should be incremented by 1 every data block.
    indices['supply_voltage'] += 1


def plural(number_of_items):
    """Utility function to pluralize words based on the number of items.
    """
    if number_of_items == 1:
        return ''
    return 's'
