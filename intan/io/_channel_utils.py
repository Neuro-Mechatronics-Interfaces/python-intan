def print_all_channel_names(result):
    """Searches through all present signal types in 'result' dict, and prints
    the names of these channels. Useful, for example, to determine names of
    channels that can be plotted.

    Parameters:
        result (dict): The result of a call to read_header() or header_to_result().
    """
    if 'amplifier_channels' in result:
        print_names_in_group(result['amplifier_channels'])

    if 'aux_input_channels' in result:
        print_names_in_group(result['aux_input_channels'])

    if 'supply_voltage_channels' in result:
        print_names_in_group(result['supply_voltage_channels'])

    if 'board_adc_channels' in result:
        print_names_in_group(result['board_adc_channels'])

    if 'board_dig_in_channels' in result:
        print_names_in_group(result['board_dig_in_channels'])

    if 'board_dig_out_channels' in result:
        print_names_in_group(result['board_dig_out_channels'])


def print_names_in_group(signal_group):
    """Searches through all channels in this group and print them.

    Parameters:
        signal_group (list): The list of channels to search through.
    """
    for this_channel in signal_group:
        print(this_channel['custom_channel_name'])


def find_channel_in_group(channel_name, signal_group):
    """Finds a channel with this name in this group, returning whether or not
    it's present and, if so, the position of this channel in signal_group.

    Parameters:
        channel_name (str): The name of the channel to search for.
        signal_group (list): The list of channels to search through.

    Returns:
        channel_found (bool): Whether or not the channel was found.
        channel_index (int): The index of the channel in signal_group.
    """
    for count, this_channel in enumerate(signal_group):
        if this_channel['custom_channel_name'] == channel_name:
            return True, count
    return False, 0


def find_channel_in_header(channel_name, header):
    """Looks through all present signal groups in header, searching for
    'channel_name'. If found, return the signal group and the index of that
    channel within the group.

    Parameters:
        channel_name (str): The name of the channel to search for.
        header (dict): The header dictionary containing signal groups.

    Returns:
        channel_found (bool): Whether or not the channel was found.
        signal_group_name (str): The name of the signal group containing the channel.
        channel_index (int): The index of the channel in the signal group.
    """
    signal_group_name = ''
    if 'amplifier_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['amplifier_channels'])
        if channel_found:
            signal_group_name = 'amplifier_channels'

    if not channel_found and 'aux_input_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['aux_input_channels'])
        if channel_found:
            signal_group_name = 'aux_input_channels'

    if not channel_found and 'supply_voltage_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['supply_voltage_channels'])
        if channel_found:
            signal_group_name = 'supply_voltage_channels'

    if not channel_found and 'board_adc_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['board_adc_channels'])
        if channel_found:
            signal_group_name = 'board_adc_channels'

    if not channel_found and 'board_dig_in_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['board_dig_in_channels'])
        if channel_found:
            signal_group_name = 'board_dig_in_channels'

    if not channel_found and 'board_dig_out_channels' in header:
        channel_found, channel_index = find_channel_in_group(
            channel_name, header['board_dig_out_channels'])
        if channel_found:
            signal_group_name = 'board_dig_out_channels'

    if channel_found:
        return True, signal_group_name, channel_index

    return False, '', 0
