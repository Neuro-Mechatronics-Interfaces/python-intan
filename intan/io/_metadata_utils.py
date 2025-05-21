"""
intan.io._metadata_utils

Utility functions for interpreting binary structure metadata in `.rhd` files.
This module calculates byte lengths, estimates the number of signal samples,
and prints summary information about recorded datasets.

Key responsibilities:
- Calculating bytes per data block for each signal type
- Estimating sample counts across channels
- Detecting file truncation or corruption based on size
- Displaying duration and sampling summary in console
"""
import os
from intan.io._exceptions import FileSizeError


# === General Utilities ===
def plural(number_of_items):
    """
    Return 's' if the number of items is not 1 (for pluralization).

    Parameters:
        number_of_items (int): Quantity to evaluate

    Returns:
        str: 's' if plural, '' if singular
    """
    return '' if number_of_items == 1 else 's'


# === Data Size and Sample Count Estimation ===
def get_bytes_per_data_block(header):
    """
    Calculate the total number of bytes in each data block of a `.rhd` file.

    This is based on the number of enabled channels and the system used for recording
    (either 60 or 128 samples per block).

    Parameters:
        header (dict): Parsed header metadata with fields like 'num_amplifier_channels'

    Returns:
        int: Number of bytes in one full data block
    """
    # Timestamps (one channel always present): Start with 4 bytes per sample.
    bytes_per_block = bytes_per_signal_type(
        header['num_samples_per_data_block'],
        1,
        4)

    # Amplifier data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'],
        header['num_amplifier_channels'],
        2)

    # Auxiliary data: Add 2 bytes per sample per enabled aux input channel.
    # Note that aux inputs are sample 4x slower than amplifiers, so there
    # are 1/4 as many samples.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'] / 4,
        header['num_aux_input_channels'],
        2)

    # Supply voltage: Add 2 bytes per sample per enabled vdd channel.
    # Note that aux inputs are sampled once per data block
    # (60x or 128x slower than amplifiers), so there are
    # 1/60 or 1/128 as many samples.
    bytes_per_block += bytes_per_signal_type(
        1,
        header['num_supply_voltage_channels'],
        2)

    # Analog inputs: Add 2 bytes per sample per enabled analog input channel.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'],
        header['num_board_adc_channels'],
        2)

    # Digital inputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            header['num_samples_per_data_block'],
            1,
            2)

    # Digital outputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            header['num_samples_per_data_block'],
            1,
            2)

    # Temp sensor: Add 2 bytes per sample per enabled temp sensor channel.
    # Note that temp sensor inputs are sampled once per data block
    # (60x or 128x slower than amplifiers), so there are
    # 1/60 or 1/128 as many samples.
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            1,
            header['num_temp_sensor_channels'],
            2)

    return bytes_per_block


def bytes_per_signal_type(num_samples, num_channels, bytes_per_sample):
    """
    Calculate number of bytes for a specific signal type in a data block.

    Parameters:
        num_samples (int or float): Samples per block (may be fractional)
        num_channels (int): Number of enabled channels for this signal type
        bytes_per_sample (int): Number of bytes per sample

    Returns:
        float: Number of bytes for this signal type
    """
    return num_samples * num_channels * bytes_per_sample


def calculate_data_size(header, filename, fid, verbose=True):
    """
    Determine the size and structure of recorded data in an `.rhd` file.

    Computes how many samples exist, how many data blocks are present,
    and whether the file appears truncated or corrupt.

    Parameters:
        header (dict): Parsed header from file
        filename (str): Path to the `.rhd` file
        fid (file): Open file object positioned after header
        verbose (bool): If True, print file duration summary

    Returns:
        tuple:
            data_present (bool): True if data exists beyond header
            filesize (int): Full file size in bytes
            num_blocks (int): Number of data blocks present
            num_samples (dict): Estimated samples per signal type
    """
    bytes_per_block = get_bytes_per_data_block(header)

    # Determine filesize and if any data is present.
    filesize = os.path.getsize(filename)
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    # If the file size is somehow different than expected, raise an error.
    if bytes_remaining % bytes_per_block != 0:
        raise FileSizeError(
            'Something is wrong with file size : '
            'should have a whole number of data blocks')

    # Calculate how many data blocks are present.
    num_blocks = int(bytes_remaining / bytes_per_block)

    num_samples = calculate_num_samples(header, num_blocks)

    if verbose:
        print_record_time_summary(num_samples['amplifier'],
                                  header['sample_rate'],
                                  data_present)

    return data_present, filesize, num_blocks, num_samples


def calculate_num_samples(header, num_data_blocks):
    """
    Estimate the number of samples for each signal type.

    Parameters:
        header (dict): Parsed header with sample structure
        num_data_blocks (int): Total data blocks in file

    Returns:
        dict: Signal type → number of samples (e.g., 'amplifier': 25600)
    """
    samples_per_block = header['num_samples_per_data_block']
    num_samples = {}
    num_samples['amplifier'] = int(samples_per_block * num_data_blocks)
    num_samples['aux_input'] = int((samples_per_block / 4) * num_data_blocks)
    num_samples['supply_voltage'] = int(num_data_blocks)
    num_samples['board_adc'] = int(samples_per_block * num_data_blocks)
    num_samples['board_dig_in'] = int(samples_per_block * num_data_blocks)
    num_samples['board_dig_out'] = int(samples_per_block * num_data_blocks)
    return num_samples


# === Console Summary ===
def print_record_time_summary(num_amp_samples, sample_rate, data_present):
    """
    Print the estimated duration of the `.rhd` recording to the console.

    Parameters:
        num_amp_samples (int): Number of amplifier samples in file
        sample_rate (float): Sampling rate in Hz
        data_present (bool): Whether any data was detected in the file
    """
    record_time = num_amp_samples / sample_rate

    if data_present:
        print('File contains {:0.3f} seconds of data.  '
              'Amplifiers were sampled at {:0.2f} kS/s.'
              .format(record_time, sample_rate / 1000))
    else:
        print('Header file contains no data.  '
              'Amplifiers were sampled at {:0.2f} kS/s.'
              .format(sample_rate / 1000))
