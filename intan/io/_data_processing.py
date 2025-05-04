import math
import numpy as np
from intan.io._file_utils import print_progress

def parse_data(header, data):
    """Parses raw data into user readable and interactable forms (for example,
    extracting raw digital data to separate channels and scaling data to units
    like microVolts, degrees Celsius, or seconds.)
    """
    print('Parsing data...')
    extract_digital_data(header, data)
    scale_analog_data(header, data)
    scale_timestamps(header, data)

def scale_timestamps(header, data):
    """Verifies no timestamps are missing, and scales timestamps to seconds.
    """
    # Check for gaps in timestamps.
    num_gaps = np.sum(np.not_equal(
        data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
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

def apply_notch_filter(header, data, verbose=True):
    """Checks header to determine if notch filter should be applied, and if so,
    apply notch filter to all signals in data['amplifier_data'].
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
            percent_done = print_progress(i+1, header['num_amplifier_channels'],
                                      print_step, percent_done)

def notch_filter(signal_in, f_sample, f_notch, bandwidth):
    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector 'signal_in'.

    f_sample = sample rate of data (input Hz or Samples/sec)
    f_notch = filter notch frequency (input Hz)
    bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(signal_in, 30000, 60, 10);
    """
    # Calculate parameters used to implement IIR filter
    t_step = 1.0/f_sample
    f_c = f_notch*t_step
    signal_length = len(signal_in)
    iir_parameters = calculate_iir_parameters(bandwidth, t_step, f_c)

    # Create empty signal_out NumPy array
    signal_out = np.zeros(signal_length)

    # Set the first 2 samples of signal_out to signal_in.
    # If filtering a continuous data stream, change signal_out[0:1] to the
    # previous final two values of signal_out
    signal_out[0] = signal_in[0]
    signal_out[1] = signal_in[1]

    # Run filter.
    for i in range(2, signal_length):
        signal_out[i] = calculate_iir(i, signal_in, signal_out, iir_parameters)

    return signal_out

def calculate_iir_parameters(bandwidth, t_step, f_c):
    """Calculates parameters d, b, a0, a1, a2, a, b0, b1, and b2 used for
    IIR filter and return them in a dict.
    """
    parameters = {}
    d = math.exp(-2.0*math.pi*(bandwidth/2.0)*t_step)
    b = (1.0 + d*d) * math.cos(2.0*math.pi*f_c)
    a0 = 1.0
    a1 = -b
    a2 = d*d
    a = (1.0 + d*d)/2.0
    b0 = 1.0
    b1 = -2.0 * math.cos(2.0*math.pi*f_c)
    b2 = 1.0

    parameters['d'] = d
    parameters['b'] = b
    parameters['a0'] = a0
    parameters['a1'] = a1
    parameters['a2'] = a2
    parameters['a'] = a
    parameters['b0'] = b0
    parameters['b1'] = b1
    parameters['b2'] = b2
    return parameters

def calculate_iir(i, signal_in, signal_out, iir_parameters):
    """Calculates a single sample of IIR filter passing signal_in through
    iir_parameters, resulting in signal_out.
    """
    sample = ((
        iir_parameters['a'] * iir_parameters['b2'] * signal_in[i - 2]
        + iir_parameters['a'] * iir_parameters['b1'] * signal_in[i - 1]
        + iir_parameters['a'] * iir_parameters['b0'] * signal_in[i]
        - iir_parameters['a2'] * signal_out[i - 2]
        - iir_parameters['a1'] * signal_out[i - 1])
        / iir_parameters['a0'])

    return sample