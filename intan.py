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

def set_channel_tcp(tcp_client, port='a', channel=0, enable_low=False, enable_high=True, enable_wide=False, enable_spike=False):
    """
    Configure a channel's TCP state for the Intan system.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.
        port (str): Port letter 'a', 'b', 'c', or 'd' (default: 'a').
        channel (int): Channel number (0 to 255).
        enable_low (bool): Enable low output (default: False).
        enable_high (bool): Enable high output (default: True).
        enable_wide (bool): Enable wide output (default: False).
        enable_spike (bool): Enable spike output (default: False).

    Raises:
        ValueError: If invalid port letter or channel number is provided.
    """
    # Validate inputs
    if port not in {'a', 'b', 'c', 'd'}:
        raise ValueError("port_letter must be one of {'a', 'b', 'c', 'd'}.")
    if not (0 <= channel <= 255):
        raise ValueError("channel_number must be between 0 and 255.")

    # Convert boolean flags to 'true' or 'false' strings
    enable_low_str = 'true' if enable_low else 'false'
    enable_high_str = 'true' if enable_high else 'false'
    enable_wide_str = 'true' if enable_wide else 'false'
    enable_spike_str = 'true' if enable_spike else 'false'

    # Format commands
    commands = [
        f"set {port_letter.lower()}-{channel_number:03d}.tcpdataoutputenabled {enable_wide_str}",
        f"set {port_letter.lower()}-{channel_number:03d}.tcpdataoutputenabledhigh {enable_high_str}",
        f"set {port_letter.lower()}-{channel_number:03d}.tcpdataoutputenabledlow {enable_low_str}",
        f"set {port_letter.lower()}-{channel_number:03d}.tcpdataoutputenabledspike {enable_spike_str}",
    ]

    # Send commands to the server
    for command in commands:
        tcp_client.sendall(command.encode('utf-8'))  # Send as UTF-8 encoded bytes

def stop_running(tcp_client):
    """
    Stops recording from the Intan RHX software by setting the run mode to 'stop'.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.
    """
    # Command to set run mode to 'stop'
    command = "set runmode stop"

    # Send the command to the server
    tcp_client.sendall(command.encode('utf-8'))

def start_running(tcp_client):
    """
    Starts recording from the Intan RHX software by setting the run mode to 'run'.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.
    """
    # Command to set run mode to 'run'
    command = "set runmode run"

    # Send the command to the server
    tcp_client.sendall(command.encode('utf-8'))

def clear_data_outputs(tcp_client):
    """
    Clear all data outputs from the Intan RHX software.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.
    """
    # Command to clear all data outputs
    command = "execute clearalldataoutputs"

    # Send the command to the server
    tcp_client.sendall(command.encode('utf-8'))

def get_sample_rate(tcp_client):
    """
    Get the current sample rate of the Intan RHX software.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.

    Returns:
        float: The current sample rate in Hz.
    """
    # Command to get the sample rate
    command = "get sampleperatehertz"

    # Send the command to the server
    resp = tcp_client.sendall(command.encode('utf-8'), wait_for_response=True).decode('utf-8')

    # Extract the sample rate from the response
    sample_rate = float(resp.split()[-1])  # Last element in the response

    return sample_rate

def set_data_blocks_per_write(tcp_client, num_blocks):
    """
    Set the number of data blocks per write for the Intan RHX software.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.
        num_blocks (int): Number of data blocks per write.
    """
    # Command to set the number of data blocks per write
    command = f"set TCPNumberDataBlocksPerWrite {num_blocks}"

    # Send the command to the server
    tcp_client.sendall(command.encode('utf-8'))

def enable_dig_in(tcp_client):
    """
    Enables and sets recording for DIGITAL-IN-1 and DIGITAL-IN-2 channels.

    Args:
        tcp_client (socket.socket): Connected TCP client socket.
    """
    # Commands to enable and set recording for DIGITAL-IN channels
    commands = [
        "set DIGITAL-IN-1.enabled true",
        "set DIGITAL-IN-1.recordingenabled true",
        "set DIGITAL-IN-2.enabled true",
        "set DIGITAL-IN-2.recordingenabled true"
    ]

    # Send each command to the server
    for command in commands:
        tcp_client.sendall(command.encode('utf-8'))

def process_waveform_block(data, num_amplifier_channels, num_analog_channels, dig_in_present, sample_rate):
    """
    Process a block of waveform data.

    Args:
        data (bytes): Raw waveform data.
        num_amplifier_channels (int): Number of amplifier channels.
        num_analog_channels (int): Number of analog channels.
        dig_in_present (bool): Whether digital input is present.
        sample_rate (float): Sampling rate.

    Returns:
        tuple: (timestamps, amplifier_data, analog_data, digital_data)
    """
    timestep = 1 / sample_rate
    num_channels = num_amplifier_channels + num_analog_channels + (1 if dig_in_present else 0)
    num_frames = len(data) // (4 + 2 * num_channels)

    timestamps = np.zeros(num_frames)
    amplifier_data = np.zeros((num_amplifier_channels, num_frames))
    analog_data = np.zeros((num_analog_channels, num_frames)) if num_analog_channels > 0 else None
    digital_data = np.zeros((1, num_frames)) if dig_in_present else None

    index = 0
    for frame in range(num_frames):
        # Read timestamp
        raw_timestamp, index = read_int32(data, index)
        timestamps[frame] = raw_timestamp * timestep

        # Read amplifier data
        for ch in range(num_amplifier_channels):
            raw_sample, index = read_uint16(data, index)
            amplifier_data[ch, frame] = SAMPLE_SCALE_FACTOR * (raw_sample - 32768)

        # Read analog data
        if num_analog_channels > 0:
            for ch in range(num_analog_channels):
                raw_sample, index = read_uint16(data, index)
                analog_data[ch, frame] = ANALOG_SCALE_FACTOR * (raw_sample - 32768)

        # Read digital data
        if dig_in_present:
            raw_sample, index = read_uint16(data, index)
            digital_data[0, frame] = raw_sample

    return timestamps, amplifier_data, analog_data, digital_data

def read_waveform_byte_block(tcp_client, num_amplifier_channels, num_analog_channels, dig_in_present, blocks_per_read=100, sample_rate=20000):
    """
    Read a waveform byte block from the TCP client.

    Args:
        tcp_client (socket.socket): Connected TCP client.
        num_amplifier_channels (int): Number of amplifier channels.
        num_analog_channels (int): Number of analog channels.
        dig_in_present (bool): Whether digital input is present.
        blocks_per_read (int): Number of blocks to read.
        sample_rate (float): Sampling rate.

    Returns:
        tuple: (timestamps, amplifier_data, analog_data, digital_data)
    """
    timestep = 1 / sample_rate
    waveform_bytes_per_frame = 4 + 2 * (num_amplifier_channels + num_analog_channels + (1 if dig_in_present else 0))
    waveform_bytes_per_block = FRAMES_PER_BLOCK * waveform_bytes_per_frame + 4
    waveform_bytes_total = blocks_per_read * waveform_bytes_per_block

    try:
        # Read data from the TCP client
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
            # Check the magic number
            magic_number, index = read_uint32(raw_data, index)
            if magic_number != MAGIC_NUMBER:
                print(f"Error: Invalid magic number in block {block}. Skipping...")
                continue

            # Extract the block data
            block_data = raw_data[index:index + (FRAMES_PER_BLOCK * waveform_bytes_per_frame)]
            index += FRAMES_PER_BLOCK * waveform_bytes_per_frame

            # Process the block data
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

