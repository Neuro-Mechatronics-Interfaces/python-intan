
"""
An unofficial Python API for the intan RHX recording system.

Contributors:
    Jonathan Shulgach (jshulgach@andrew.cmu.edu)
    Max Murphy (mdmurphy@andrew.cmu.edu)

Last Modified: 12/29/2024
"""
__author__ = 'Jonathan Shulgach'
__version__ = '0.0.1'

import os
import numpy as np

MAGIC_NUMBER = 0x2ef07a08
FRAMES_PER_BLOCK = 128  # 128 frames per block
SAMPLE_SCALE_FACTOR = 0.195  # Scale factor for amplifier data
ANALOG_SCALE_FACTOR = 312.5e-6  # Scale factor for analog data

# Byte Parsing Functions
def read_uint32(array, arrayIndex):
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    return variable, arrayIndex + 4

def read_int32(array, arrayIndex):
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    return variable, arrayIndex + 4

def read_uint16(array, arrayIndex):
    variableBytes = array[arrayIndex: arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    return variable, arrayIndex + 2

def set_channel_tcp(tcp_client, port='a', channel=0, enable_low=False, enable_high=True, enable_wide=False, enable_spike=False):
    """
    Sets the TCP data output for a specific channel on the Intan RHX system.

    :param tcp_client:    TCP client object for communication with the Intan RHX system.
    :param port:          Port letter ('a', 'b', 'c', or 'd') to specify the channel.
    :param channel:       Channel number (0-255) to be configured.
    :param enable_low:    Enable low frequency data output.
    :param enable_high:   Enable high frequency data output.
    :param enable_wide:   Enable wideband data output.
    :param enable_spike:  Enable spike data output.
    :return:
    """
    if port not in {'a', 'b', 'c', 'd'}:
        raise ValueError("port_letter must be one of {'a', 'b', 'c', 'd'}.")
    if not (0 <= channel <= 255):
        raise ValueError("channel_number must be between 0 and 255.")

    enable_low_str = 'true' if enable_low else 'false'
    enable_high_str = 'true' if enable_high else 'false'
    enable_wide_str = 'true' if enable_wide else 'false'
    enable_spike_str = 'true' if enable_spike else 'false'

    commands = [
        f"set {port}-{channel:03d}.tcpdataoutputenabled {enable_wide_str}",
        f"set {port}-{channel:03d}.tcpdataoutputenabledhigh {enable_high_str}",
        f"set {port}-{channel:03d}.tcpdataoutputenabledlow {enable_low_str}",
        f"set {port}-{channel:03d}.tcpdataoutputenabledspike {enable_spike_str}",
    ]

    for command in commands:
        tcp_client.sendall(command.encode('utf-8'))

def stop_running(tcp_client):
    """
    Stops recording from the Intan RHX software by setting the run mode to 'stop'.
    :param tcp_client:
    :return:
    """
    tcp_client.sendall(b"set runmode stop")

def start_running(tcp_client):
    """
    Starts recording from the Intan RHX software by setting the run mode to 'run'.
    :param tcp_client:
    :return:
    """
    tcp_client.sendall(b"set runmode run")

def clear_data_outputs(tcp_client):
    """
    Clears all data outputs from the Intan RHX software.
    :param tcp_client:
    :return:
    """
    tcp_client.sendall(b"execute clearalldataoutputs")

def get_sample_rate(tcp_client):
    """
    Gets the sample rate from the Intan RHX software.
    :param tcp_client:
    :return:
    """
    command = "get sampleperatehertz"
    tcp_client.sendall(command.encode('utf-8'))
    resp = tcp_client.recv(1024).decode('utf-8')
    return float(resp.split()[-1])

def set_data_blocks_per_write(tcp_client, num_blocks):
    """
    Sets the number of data blocks per write for the Intan RHX software.
    :param tcp_client:
    :param num_blocks:
    :return:
    """
    command = f"set TCPNumberDataBlocksPerWrite {num_blocks}"
    tcp_client.sendall(command.encode('utf-8'))

def enable_dig_in(tcp_client, dig_input=1):
    """
    Enables digital input channel for recording.
    :param tcp_client:    TCP client object for communication with the Intan RHX system.
    :param dig_input:     Digital input channel number (1-4).
    :return:
    """
    commands = [
        f"set DIGITAL-IN-{str(dig_input)}.enabled true",
        f"set DIGITAL-IN-{str(dig_input)}.recordingenabled true"
    ]
    for command in commands:
        tcp_client.sendall(command.encode('utf-8'))

def disable_dig_in(tcp_client, dig_input=1):
    """
    Disables digital input channel for recording.
    :param tcp_client:    TCP client object for communication with the Intan RHX system.
    :param dig_input:     Digital input channel number (1-4).
    :return:
    """
    commands = [
        f"set DIGITAL-IN-{str(dig_input)}.enabled false",
        f"set DIGITAL-IN-{str(dig_input)}.recordingenabled false"
    ]
    for command in commands:
        tcp_client.sendall(command.encode('utf-8'))

def process_waveform_block(data, num_amplifier_channels, num_analog_channels, dig_in_present, sample_rate):
    timestep = 1 / sample_rate
    num_channels = num_amplifier_channels + num_analog_channels + (1 if dig_in_present else 0)
    num_frames = len(data) // (4 + 2 * num_channels)

    timestamps = np.zeros(num_frames)
    amplifier_data = np.zeros((num_amplifier_channels, num_frames))
    analog_data = np.zeros((num_analog_channels, num_frames)) if num_analog_channels > 0 else None
    digital_data = np.zeros((1, num_frames)) if dig_in_present else None

    index = 0
    for frame in range(num_frames):
        raw_timestamp, index = read_int32(data, index)
        timestamps[frame] = raw_timestamp * timestep

        for ch in range(num_amplifier_channels):
            raw_sample, index = read_uint16(data, index)
            amplifier_data[ch, frame] = SAMPLE_SCALE_FACTOR * (raw_sample - 32768)

        if num_analog_channels > 0:
            for ch in range(num_analog_channels):
                raw_sample, index = read_uint16(data, index)
                analog_data[ch, frame] = ANALOG_SCALE_FACTOR * (raw_sample - 32768)

        if dig_in_present:
            raw_sample, index = read_uint16(data, index)
            digital_data[0, frame] = raw_sample

    return timestamps, amplifier_data, analog_data, digital_data

def read_waveform_byte_block(tcp_client, num_amplifier_channels, num_analog_channels, dig_in_present, blocks_per_read=100, sample_rate=20000):
    timestep = 1 / sample_rate
    waveform_bytes_per_frame = 4 + 2 * (num_amplifier_channels + num_analog_channels + (1 if dig_in_present else 0))
    waveform_bytes_per_block = FRAMES_PER_BLOCK * waveform_bytes_per_frame + 4
    waveform_bytes_total = blocks_per_read * waveform_bytes_per_block

    try:
        raw_data = tcp_client.recv(waveform_bytes_total)
        if len(raw_data) < waveform_bytes_total:
            print("Insufficient data received.")
            return None, None, None, None

        timestamps = []
        amplifier_data = []
        analog_data = []
        digital_data = []
        index = 0

        for block in range(blocks_per_read):
            magic_number, index = read_uint32(raw_data, index)
            if magic_number != MAGIC_NUMBER:
                print(f"Error: Invalid magic number in block {block}. Skipping...")
                continue

            block_data = raw_data[index:index + (FRAMES_PER_BLOCK * waveform_bytes_per_frame)]
            index += FRAMES_PER_BLOCK * waveform_bytes_per_frame

            ts, amp_data, ana_data, dig_data = process_waveform_block(
                block_data, num_amplifier_channels, num_analog_channels, dig_in_present, sample_rate
            )
            timestamps.extend(ts)
            amplifier_data.append(amp_data)
            if ana_data is not None:
                analog_data.append(ana_data)
            if dig_data is not None:
                digital_data.append(dig_data)

        return (
            np.array(timestamps),
            np.concatenate(amplifier_data, axis=1),
            np.concatenate(analog_data, axis=1) if analog_data else None,
            np.concatenate(digital_data, axis=1) if digital_data else None,
        )

    except Exception as e:
        print(f"Error reading waveform block: {e}")
        return None, None, None, None
