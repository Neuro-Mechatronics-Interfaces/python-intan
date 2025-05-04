
def plural(number_of_items):
    """Utility function to pluralize words based on the number of items.
    """
    if number_of_items == 1:
        return ''
    return 's'

def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60 or 128 sample datablock."""
    # Depending on the system used to acquire the data,
    # 'num_samples_per_data_block' will be either 60 (USB Interface Board)
    # or 128 (Recording Controller).
    # Use this number along with numbers of channels to accrue a sum of how
    # many bytes each data block should contain.

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
    """Calculates the number of bytes, per data block, for a signal type
    provided the number of samples (per data block), the number of enabled
    channels, and the size of each sample in bytes.
    """
    return num_samples * num_channels * bytes_per_sample

def calculate_data_size(header, filename, fid, verbose=True):
    """Calculates how much data is present in this file. Returns:
    data_present: Bool, whether any data is present in file
    filesize: Int, size (in bytes) of file
    num_blocks: Int, number of 60 or 128-sample data blocks present
    num_samples: Int, number of samples present in file
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
    """Calculates number of samples for each signal type, storing the results
    in num_samples dict for later use.
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

def print_record_time_summary(num_amp_samples, sample_rate, data_present):
    """Prints summary of how much recorded data is present in RHD file
    to console.
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