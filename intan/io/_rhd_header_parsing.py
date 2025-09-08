"""
intan.io._header_parsing

Low-level parser for extracting metadata and channel structure from `.rhd`
files recorded by Intan Technologies hardware.

This module reads:
- File version and magic number
- Signal groups and their channel maps
- Amplifier settings and frequency parameters
- Qt-style strings and notes
- Impedance settings, board configuration, and reference info

Primary function:
    - `read_header(fid)`: returns a fully populated header dictionary

Used internally by `intan.io._rhd_loader` to build a unified `result` dictionary
for EMG/LFP signal analysis.
"""
import os
import struct
from intan.io._exceptions import UnknownChannelTypeError, QStringError, UnrecognizedFileError, FileSizeError


# === Public API ===

def read_header(fid):
    """"
    Parse the binary file header from an Intan `.rhd` data file.

    This function checks the magic number, reads the file version, evaluates
    signal settings, channel layouts, impedance settings, and any embedded notes.

    Parameters:
        fid (file): Opened file object positioned at the start of the file.

    Returns:
        dict: Parsed header metadata.
    """
    check_magic_number(fid)

    header = {}

    read_version_number(header, fid)
    set_num_samples_per_data_block(header)

    freq = {}

    read_sample_rate(header, fid)
    read_freq_settings(freq, fid)
    read_notch_filter_frequency(header, freq, fid)
    read_impedance_test_frequencies(freq, fid)
    read_notes(header, fid)
    read_num_temp_sensor_channels(header, fid)
    read_eval_board_mode(header, fid)
    read_reference_channel(header, fid)

    set_sample_rates(header, freq)
    set_frequency_parameters(header, freq)

    initialize_channels(header)

    read_channel_structure(header, fid)

    return header


# === Low-Level Binary Readers ===

def read_qstring(fid):
    """
    Read a QString (Unicode) from a Qt-generated binary file.

    Format:
    - First 4 bytes: length in bytes (uint32)
    - If 0xFFFFFFFF, return empty string
    - Content is 16-bit unicode characters

    Parameters:
        fid (file): File object positioned at the string start.

    Returns:
        str: Decoded unicode string.

    Raises:
        QStringError: If the declared length exceeds file size.
    """
    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16):
        return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1):
        print(length)
        raise QStringError('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for _ in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    return ''.join([chr(c) for c in data])


def read_notes(header, fid):
    """Reads notes as QStrings from fid, and stores them as strings in
    header['notes'] dict.

    Parameters:
        header (dict): Header dictionary to store notes.
        fid (file): Opened file object positioned at the start of the file.
    """
    header['notes'] = {'note1': read_qstring(fid),
                       'note2': read_qstring(fid),
                       'note3': read_qstring(fid)}


# === File Formet Decoding ===

def read_version_number(header, fid, verbose=True):
    """Reads version number (major and minor) from fid. Stores them into
    header['version']['major'] and header['version']['minor'].

    Parameters:
        header (dict): Header dictionary to store version information.
        fid (file): Opened file object positioned at the start of the file.
        verbose (bool): If True, print version information to console.

    Raises:
        UnrecognizedFileError: If the magic number does not match the expected
    """
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    if verbose:
        print('\nReading Intan Technologies RHD Data File, Version {}.{}\n'
              .format(version['major'], version['minor']))


def check_magic_number(fid):
    """Checks magic number at beginning of file to verify this is an Intan
    Technologies RHD data file.

    Parameters:
        fid (file): Opened file object positioned at the start of the file.

    Raises:
        UnrecognizedFileError: If the magic number does not match the expected
    """
    magic_number, = struct.unpack('<I', fid.read(4))
    if magic_number != int('c6912702', 16):
        raise UnrecognizedFileError('Unrecognized file type.')


# === Channel Map Readers ===

def read_notch_filter_frequency(header, freq, fid):
    """
    Reads notch filter mode from fid, and stores frequency (in Hz) in
    'header' and 'freq' dicts.

    Parameters:
        header (dict): Header dictionary to store notch filter frequency.
        freq (dict): Dictionary to store notch filter frequency.
        fid (file): Opened file object positioned at the start of the file.
    """
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']


def read_channel_structure(header, fid, verbose=False):
    """
    Reads signal summary from data file header and stores information for
    all signal groups and their channels in 'header' dict.

    Parameters:
        header (dict): Header dictionary to store channel information.
        fid (file): Opened file object positioned at the start of the file.
        verbose (bool): If True, print header summary to console.
    """
    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    for signal_group in range(1, number_of_signal_groups + 1):
        add_signal_group_information(header, fid, signal_group)
    add_num_channels(header)
    if verbose:
        print_header_summary(header)


def read_num_temp_sensor_channels(header, fid):
    """
    Stores number of temp sensor channels in
    header['num_temp_sensor_channels']. Temp sensor data may be saved from
    versions 1.1 and later.

    Parameters:
        header (dict): Header dictionary to store temp sensor channel count.
        fid (file): Opened file object positioned at the start of the file.
    """
    header['num_temp_sensor_channels'] = 0
    if ((header['version']['major'] == 1 and header['version']['minor'] >= 1)
            or (header['version']['major'] > 1)):
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))


def read_sample_rate(header, fid):
    """
    Reads sample rate from fid. Stores it into header['sample_rate'].

    Parameters:
        header (dict): Header dictionary to store sample rate.
        fid (file): Opened file object positioned at the start of the file.
    """
    header['sample_rate'], = struct.unpack('<f', fid.read(4))


def read_freq_settings(freq, fid):
    """
    Reads amplifier frequency settings from fid. Stores them in 'freq' dict.

    Parameters:
        freq (dict): Dictionary to store frequency settings.
        fid (file): Opened file object positioned at the start of the file.
    """
    (freq['dsp_enabled'],
     freq['actual_dsp_cutoff_frequency'],
     freq['actual_lower_bandwidth'],
     freq['actual_upper_bandwidth'],
     freq['desired_dsp_cutoff_frequency'],
     freq['desired_lower_bandwidth'],
     freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))


def read_impedance_test_frequencies(freq, fid):
    """
    Reads desired and actual impedance test frequencies from fid, and stores
    them (in Hz) in 'freq' dicts.

    Parameters:
        freq (dict): Dictionary to store impedance test frequencies.
        fid (file): Opened file object positioned at the start of the file.
    """
    (freq['desired_impedance_test_frequency'],
     freq['actual_impedance_test_frequency']) = (
        struct.unpack('<ff', fid.read(8)))


def read_eval_board_mode(header, fid):
    """
    Stores eval board mode in header['eval_board_mode']. Board mode is saved
    from versions 1.3 and later.

    Parameters:
        header (dict): Header dictionary to store eval board mode.
        fid (file): Opened file object positioned at the start of the file.
    """
    header['eval_board_mode'] = 0
    if ((header['version']['major'] == 1 and header['version']['minor'] >= 3)
            or (header['version']['major'] > 1)):
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))


def read_reference_channel(header, fid):
    """
    Reads name of reference channel as QString from fid, and stores it as
    a string in header['reference_channel']. Data files v2.0 or later include
    reference channel.

    Parameters:
        header (dict): Header dictionary to store reference channel name.
        fid (file): Opened file object positioned at the start of the file.
    """
    if header['version']['major'] > 1:
        header['reference_channel'] = read_qstring(fid)


def read_new_channel(fid, signal_group_name, signal_group_prefix,
                     signal_group):
    """
    Reads a new channel's information from fid and returns it as a dict.
    The channel is identified by its signal group name, prefix, and number.

    Parameters:
        fid (file): Opened file object positioned at the start of the channel.
        signal_group_name (str): Name of the signal group.
        signal_group_prefix (str): Prefix of the signal group.
        signal_group (int): Number of the signal group.

    Returns:
       new_channel (dict): Dictionary with channel information.
       new_trigger_channel (dict): Dictionary with trigger channel info.
       channel_enabled (bool): Indicates if the channel is enabled.
       signal_type (int): Type of signal for the channel.
    """
    new_channel = {'port_name': signal_group_name,
                   'port_prefix': signal_group_prefix,
                   'port_number': signal_group}
    new_channel['native_channel_name'] = read_qstring(fid)
    new_channel['custom_channel_name'] = read_qstring(fid)
    (new_channel['native_order'],
     new_channel['custom_order'],
     signal_type, channel_enabled,
     new_channel['chip_channel'],
     new_channel['board_stream']) = (
        struct.unpack('<hhhhhh', fid.read(12)))
    new_trigger_channel = {}
    (new_trigger_channel['voltage_trigger_mode'],
     new_trigger_channel['voltage_threshold'],
     new_trigger_channel['digital_trigger_channel'],
     new_trigger_channel['digital_edge_polarity']) = (
        struct.unpack('<hhhh', fid.read(8)))
    (new_channel['electrode_impedance_magnitude'],
     new_channel['electrode_impedance_phase']) = (
        struct.unpack('<ff', fid.read(8)))

    return new_channel, new_trigger_channel, channel_enabled, signal_type


def append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type):
    """
    Appends 'new_channel' to 'header' dict depending on if channel is
    enabled and the signal type.

    Parameters:
        header (dict): Header dictionary to store channel information.
        new_channel (dict): Dictionary with new channel information.
        new_trigger_channel (dict): Dictionary with trigger channel info.
        channel_enabled (bool): Indicates if the channel is enabled.
        signal_type (int): Type of signal for the channel.

    Raises:
        UnknownChannelTypeError: If the signal type is unrecognized.
    """
    if not channel_enabled:
        return

    if signal_type == 0:
        header['amplifier_channels'].append(new_channel)
        header['spike_triggers'].append(new_trigger_channel)
    elif signal_type == 1:
        header['aux_input_channels'].append(new_channel)
    elif signal_type == 2:
        header['supply_voltage_channels'].append(new_channel)
    elif signal_type == 3:
        header['board_adc_channels'].append(new_channel)
    elif signal_type == 4:
        header['board_dig_in_channels'].append(new_channel)
    elif signal_type == 5:
        header['board_dig_out_channels'].append(new_channel)
    else:
        raise UnknownChannelTypeError('Unknown channel type.')


def add_num_channels(header):
    """Adds channel numbers for all signal types to 'header' dict.

    Parameters:
        header (dict): Header dictionary to store channel counts.

    """
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(
        header['supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(
        header['board_dig_out_channels'])


def set_num_samples_per_data_block(header):
    """
    Determines how many samples are present per data block (60 or 128),
    depending on version. Data files v2.0 or later have 128 samples per block,
    otherwise 60.

    Parameters:
        header (dict): Header dictionary to store number of samples per block.
    """
    header['num_samples_per_data_block'] = 60
    if header['version']['major'] > 1:
        header['num_samples_per_data_block'] = 128


def set_sample_rates(header, freq):
    """Determines what the sample rates are for various signal types, and
    stores them in 'freq' dict.

    Parameters:
        header (dict): Header dictionary to store sample rates.
        freq (dict): Dictionary to store frequency parameters.
    """
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['aux_input_sample_rate'] = header['sample_rate'] / 4
    freq['supply_voltage_sample_rate'] = (header['sample_rate'] /
                                          header['num_samples_per_data_block'])
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']


def set_frequency_parameters(header, freq):
    """Stores frequency parameters (set in other functions) in
    header['frequency_parameters']

    Parameters:
        header (dict): Header dictionary to store frequency parameters.
        freq (dict): Dictionary to store frequency parameters.
    """
    header['frequency_parameters'] = freq


def initialize_channels(header):
    """Creates empty lists for each type of data channel and stores them in
    'header' dict.

    Parameters:
        header (dict): Header dictionary to initialize channel lists.

    """
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []


def add_signal_group_information(header, fid, signal_group):
    """Adds information for a signal group and all its channels to 'header'
    dict.

    Parameters:
        header (dict): Header dictionary to store signal group information.
        fid (file): Opened file object positioned at the start of the signal
                    group.
        signal_group (int): Number of the signal group.
    """
    signal_group_name = read_qstring(fid)
    signal_group_prefix = read_qstring(fid)
    (signal_group_enabled, signal_group_num_channels, _) = struct.unpack(
        '<hhh', fid.read(6))

    if signal_group_num_channels > 0 and signal_group_enabled > 0:
        for _ in range(0, signal_group_num_channels):
            add_channel_information(header, fid, signal_group_name,
                                    signal_group_prefix, signal_group)


def add_channel_information(header, fid, signal_group_name,
                            signal_group_prefix, signal_group):
    """Reads a new channel's information from fid and appends it to 'header'
    dict.

    Parameters:
        header (dict): Header dictionary to store channel information.
        fid (file): Opened file object positioned at the start of the channel.
        signal_group_name (str): Name of the signal group.
        signal_group_prefix (str): Prefix of the signal group.
        signal_group (int): Number of the signal group.
    """
    (new_channel, new_trigger_channel, channel_enabled,
     signal_type) = read_new_channel(
        fid, signal_group_name, signal_group_prefix, signal_group)
    append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type)


# === Data Post-Processing ===

def header_to_result(header, result):
    """
    Merge parsed header fields into the global `result` dictionary.

    Parameters:
        header (dict): Parsed header metadata from `read_header`.
        result (dict): Destination dictionary to be populated.

    Returns:
        dict: Updated result dictionary with signal channel mappings.
    """
    if header['num_amplifier_channels'] > 0:
        result['spike_triggers'] = header['spike_triggers']
        result['amplifier_channels'] = header['amplifier_channels']

    result['notes'] = header['notes']
    result['frequency_parameters'] = header['frequency_parameters']

    if header['version']['major'] > 1:
        result['reference_channel'] = header['reference_channel']

    if header['num_aux_input_channels'] > 0:
        result['aux_input_channels'] = header['aux_input_channels']

    if header['num_supply_voltage_channels'] > 0:
        result['supply_voltage_channels'] = header['supply_voltage_channels']

    if header['num_board_adc_channels'] > 0:
        result['board_adc_channels'] = header['board_adc_channels']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_channels'] = header['board_dig_in_channels']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_channels'] = header['board_dig_out_channels']

    return result


def print_header_summary(header):
    """
    Prints summary of contents of RHD header to console.

    Parameters:
        header (dict): Header dictionary containing parsed metadata.
    """
    print('Found {} amplifier channel{}.'.format(
        header['num_amplifier_channels'],
        plural(header['num_amplifier_channels'])))
    print('Found {} auxiliary input channel{}.'.format(
        header['num_aux_input_channels'],
        plural(header['num_aux_input_channels'])))
    print('Found {} supply voltage channel{}.'.format(
        header['num_supply_voltage_channels'],
        plural(header['num_supply_voltage_channels'])))
    print('Found {} board ADC channel{}.'.format(
        header['num_board_adc_channels'],
        plural(header['num_board_adc_channels'])))
    print('Found {} board digital input channel{}.'.format(
        header['num_board_dig_in_channels'],
        plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(
        header['num_board_dig_out_channels'],
        plural(header['num_board_dig_out_channels'])))
    print('Found {} temperature sensors channel{}.'.format(
        header['num_temp_sensor_channels'],
        plural(header['num_temp_sensor_channels'])))
    print('')


def data_to_result(header, data, result):
    """
    Merges data from all present signals into a common 'result' dict. If
    any signal types have been allocated but aren't relevant (for example,
    no channels of this type exist), does not copy those entries into 'result'.

    Parameters:
        header (dict): Parsed header metadata from `read_header`.
        data (dict): Dictionary containing signal data.
        result (dict): Destination dictionary to be populated.

    Returns:
        dict: Updated result dictionary with signal data.
    """
    if len(header['amplifier_channels']) > 0:
        result['t_amplifier'] = data['t_amplifier']
        result['amplifier_data'] = data['amplifier_data']

    if header['num_aux_input_channels'] > 0:
        result['t_aux_input'] = data['t_aux_input']
        result['aux_input_data'] = data['aux_input_data']

    if header['num_supply_voltage_channels'] > 0:
        result['t_supply_voltage'] = data['t_supply_voltage']
        result['supply_voltage_data'] = data['supply_voltage_data']

    if header['num_temp_sensor_channels'] > 0:
        result['t_temp_sensor'] = data['t_temp_sensor']

    if header['num_board_adc_channels'] > 0:
        result['t_board_adc'] = data['t_board_adc']
        result['board_adc_data'] = data['board_adc_data']

    if (header['num_board_dig_in_channels'] > 0
            or header['num_board_dig_out_channels'] > 0):
        result['t_dig'] = data['t_dig']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_data'] = data['board_dig_in_data']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_data'] = data['board_dig_out_data']

    return result

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
