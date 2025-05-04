import os
import struct
from intan.io._exceptions import UnknownChannelTypeError, QStringError, UnrecognizedFileError
from intan.io._block_parser import plural

def check_magic_number(fid):
    """Checks magic number at beginning of file to verify this is an Intan
    Technologies RHD data file.
    """
    magic_number, = struct.unpack('<I', fid.read(4))
    if magic_number != int('c6912702', 16):
        raise UnrecognizedFileError('Unrecognized file type.')

def read_notch_filter_frequency(header, freq, fid):
    """Reads notch filter mode from fid, and stores frequency (in Hz) in
    'header' and 'freq' dicts.
    """
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']

def read_header(fid):
    """Reads the Intan File Format header from the given file.
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

    read_signal_summary(header, fid)

    return header

def read_num_temp_sensor_channels(header, fid):
    """Stores number of temp sensor channels in
    header['num_temp_sensor_channels']. Temp sensor data may be saved from
    versions 1.1 and later.
    """
    header['num_temp_sensor_channels'] = 0
    if ((header['version']['major'] == 1 and header['version']['minor'] >= 1)
            or (header['version']['major'] > 1)):
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))

def read_version_number(header, fid, verbose=True):
    """Reads version number (major and minor) from fid. Stores them into
    header['version']['major'] and header['version']['minor'].
    """
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    if verbose:
        print('\nReading Intan Technologies RHD Data File, Version {}.{}\n'
          .format(version['major'], version['minor']))

def set_num_samples_per_data_block(header):
    """Determines how many samples are present per data block (60 or 128),
    depending on version. Data files v2.0 or later have 128 samples per block,
    otherwise 60.
    """
    header['num_samples_per_data_block'] = 60
    if header['version']['major'] > 1:
        header['num_samples_per_data_block'] = 128

def read_sample_rate(header, fid):
    """Reads sample rate from fid. Stores it into header['sample_rate'].
    """
    header['sample_rate'], = struct.unpack('<f', fid.read(4))

def read_freq_settings(freq, fid):
    """Reads amplifier frequency settings from fid. Stores them in 'freq' dict.
    """
    (freq['dsp_enabled'],
     freq['actual_dsp_cutoff_frequency'],
     freq['actual_lower_bandwidth'],
     freq['actual_upper_bandwidth'],
     freq['desired_dsp_cutoff_frequency'],
     freq['desired_lower_bandwidth'],
     freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))

def read_impedance_test_frequencies(freq, fid):
    """Reads desired and actual impedance test frequencies from fid, and stores
    them (in Hz) in 'freq' dicts.
    """
    (freq['desired_impedance_test_frequency'],
     freq['actual_impedance_test_frequency']) = (
         struct.unpack('<ff', fid.read(8)))

def read_notes(header, fid):
    """Reads notes as QStrings from fid, and stores them as strings in
    header['notes'] dict.
    """
    header['notes'] = {'note1': read_qstring(fid),
                       'note2': read_qstring(fid),
                       'note3': read_qstring(fid)}

def read_eval_board_mode(header, fid):
    """Stores eval board mode in header['eval_board_mode']. Board mode is saved
    from versions 1.3 and later.
    """
    header['eval_board_mode'] = 0
    if ((header['version']['major'] == 1 and header['version']['minor'] >= 3)
            or (header['version']['major'] > 1)):
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))

def read_reference_channel(header, fid):
    """Reads name of reference channel as QString from fid, and stores it as
    a string in header['reference_channel']. Data files v2.0 or later include
    reference channel.
    """
    if header['version']['major'] > 1:
        header['reference_channel'] = read_qstring(fid)

def set_sample_rates(header, freq):
    """Determines what the sample rates are for various signal types, and
    stores them in 'freq' dict.
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
    """
    header['frequency_parameters'] = freq

def initialize_channels(header):
    """Creates empty lists for each type of data channel and stores them in
    'header' dict.
    """
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

def read_signal_summary(header, fid, verbose=False):
    """Reads signal summary from data file header and stores information for
    all signal groups and their channels in 'header' dict.
    """
    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    for signal_group in range(1, number_of_signal_groups + 1):
        add_signal_group_information(header, fid, signal_group)
    add_num_channels(header)
    if verbose:
        print_header_summary(header)

def add_signal_group_information(header, fid, signal_group):
    """Adds information for a signal group and all its channels to 'header'
    dict.
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
    """
    (new_channel, new_trigger_channel, channel_enabled,
     signal_type) = read_new_channel(
         fid, signal_group_name, signal_group_prefix, signal_group)
    append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type)

def read_new_channel(fid, signal_group_name, signal_group_prefix,
                     signal_group):
    """Reads a new channel's information from fid.
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
    """"Appends 'new_channel' to 'header' dict depending on if channel is
    enabled and the signal type.
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
    """
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(
        header['supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(
        header['board_dig_out_channels'])

def header_to_result(header, result):
    """Merges header information from .rhd file into a common 'result' dict.
    If any fields have been allocated but aren't relevant (for example, no
    channels of this type exist), does not copy those entries into 'result'.
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
    """Prints summary of contents of RHD header to console.
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

def read_qstring(fid):
    """Reads Qt style QString.

    The first 32-bit unsigned number indicates the length of the string
    (in bytes). If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
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

def data_to_result(header, data, result):
    """Merges data from all present signals into a common 'result' dict. If
    any signal types have been allocated but aren't relevant (for example,
    no channels of this type exist), does not copy those entries into 'result'.
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
