#! /bin/env python3
# Adrian Foy September 2023

"""Example demonstrating reading of 1 second waveform data (wideband amplifier
data on channel A-010) using TCP command socket to control RHX software and TCP
waveform socket to read amplifier data.

In order to run this example script successfully, the Intan RHX software
should first be started, and through Network -> Remote TCP Control.

Command Output should open a connection at 127.0.0.1, Port 5000.
Status should read "Pending".

Waveform Output (in the Data Output tab) should open a connection at 127.0.0.1,
Port 5001. Status should read "Pending" for the Waveform Port (Spike Port is
unused for this example, and can be left disconnected).

Once these ports are opened, this script can be run to acquire ~1 second of
wideband data from channel A-010, which can then be plotted assuming
"matplotlib" is installed.
"""

import time
import socket

# In order to plot the data, 'matplotlib' is required.
# If plotting is not needed, calls to plt can be removed and the data
# will still be present within the ReadWaveformDataDemo() function.
# 'matplotlib' can be installed with the command 'pip install matplotlib'
import matplotlib.pyplot as plt


def readUint32(array, arrayIndex):
    """Reads 4 bytes from array as unsigned 32-bit integer.
    """
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex


def readInt32(array, arrayIndex):
    """Reads 4 bytes from array as signed 32-bit integer.
    """
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex


def readUint16(array, arrayIndex):
    """Reads 2 bytes from array as unsigned 16-bit integer.
    """
    variableBytes = array[arrayIndex: arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 2
    return variable, arrayIndex


def ReadWaveformDataDemo():
    """Read Waveform Data Demo.

    Uses TCP to control RHX software and read 1 second of waveform data,
    as a demonstration of TCP control and TCP data streaming, both of which
    are described in 'IntanRHX_TCPDocumentation.pdf'
    """

    # Connect to TCP command server - default home IP address at port 5000.
    print('Connecting to TCP command server...')
    scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scommand.connect(('127.0.0.1', 5000))

    # Connect to TCP waveform server - default home IP address at port 5001.
    print('Connecting to TCP waveform server...')
    swaveform = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    swaveform.connect(('127.0.0.1', 5001))

    # Query runmode from RHX software.
    scommand.sendall(b'get runmode')
    commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")

    # If controller is running, stop it.
    if commandReturn != "Return: RunMode Stop":
        scommand.sendall(b'set runmode stop')
        # Allow time for RHX software to accept this command before the next.
        time.sleep(0.1)

    # Query sample rate from RHX software.
    scommand.sendall(b'get sampleratehertz')
    commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
    expectedReturnString = "Return: SampleRateHertz "
    # Look for "Return: SampleRateHertz N" where N is the sample rate.
    if commandReturn.find(expectedReturnString) == -1:
        raise GetSampleRateFailure(
            'Unable to get sample rate from server.'
        )

    # Calculate timestep from sample rate.
    timestep = 1 / float(commandReturn[len(expectedReturnString):])

    # Clear TCP data output to ensure no TCP channels are enabled.
    scommand.sendall(b'execute clearalldataoutputs')
    time.sleep(0.1)

    # Send TCP commands to set up TCP Data Output Enabled for wide
    # band of channel A-010.
    scommand.sendall(b'set a-010.tcpdataoutputenabled true')
    time.sleep(0.1)

    # Calculations for accurate parsing
    # At 30 kHz with 1 channel, 1 second of wideband waveform data
    # (including magic number, timestamps, amplifier data) is 181,420 bytes.
    # N = (FRAMES_PER_BLOCK * waveformBytesPerFrame + SizeOfMagicNumber) *
    # NumBlocks where:
    # FRAMES_PER_BLOCK = 128 ; standard data block size used by Intan

    # waveformBytesPerFrame = SizeOfTimestamp + SizeOfSample ;
    # timestamp is a 4-byte int, and amplifier sample is a 2-byte unsigned int

    # SizeOfMagicNumber = 4; Magic number is a 4-byte (32-bit) unsigned int
    # NumBlocks = NumFrames / FRAMES_PER_BLOCK ;
    # At 30 kHz, 1 second of data has 30000 frames.
    # NumBlocks must be an integer value, so round up to 235
    waveformBytesPerFrame = 4 + 2
    waveformBytesPerBlock = FRAMES_PER_BLOCK * waveformBytesPerFrame + 4

    # Run controller for 1 second
    scommand.sendall(b'set runmode run')
    time.sleep(1)
    scommand.sendall(b'set runmode stop')

    # Read waveform data
    rawData = swaveform.recv(WAVEFORM_BUFFER_SIZE)
    if len(rawData) % waveformBytesPerBlock != 0:
        raise InvalidReceivedDataSize(
            'An unexpected amount of data arrived that is not an integer '
            'multiple of the expected data size per block.'
        )
    numBlocks = int(len(rawData) / waveformBytesPerBlock)

    # Index used to read the raw data that came in through the TCP socket.
    rawIndex = 0

    # List used to contain scaled timestamp values in seconds.
    amplifierTimestamps = []

    # List used to contain scaled amplifier data in microVolts.
    amplifierData = []

    for _ in range(numBlocks):
        # Expect 4 bytes to be TCP Magic Number as uint32.
        # If not what's expected, raise an exception.
        magicNumber, rawIndex = readUint32(rawData, rawIndex)
        if magicNumber != 0x2ef07a08:
            raise InvalidMagicNumber('Error... magic number incorrect')

        # Each block should contain 128 frames of data - process each
        # of these one-by-one
        for _ in range(FRAMES_PER_BLOCK):
            # Expect 4 bytes to be timestamp as int32.
            rawTimestamp, rawIndex = readInt32(rawData, rawIndex)

            # Multiply by 'timestep' to convert timestamp to seconds
            amplifierTimestamps.append(rawTimestamp * timestep)

            # Expect 2 bytes of wideband data.
            rawSample, rawIndex = readUint16(rawData, rawIndex)

            # Scale this sample to convert to microVolts
            amplifierData.append(0.195 * (rawSample - 32768))

    # If using matplotlib to plot is not desired,
    # the following plot lines can be removed.
    # Data is still accessible at this point in the amplifierTimestamps
    # and amplifierData.
    plt.plot(amplifierTimestamps, amplifierData)
    plt.title('A-010 Amplifier Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.show()


class GetSampleRateFailure(Exception):
    """Exception returned when the TCP socket failed to yield the sample rate
    as reported by the RHX software.
    """


class InvalidReceivedDataSize(Exception):
    """Exception returned when the amount of data received on the TCP socket
    is not an integer multiple of the excepted data block size.
    """


class InvalidMagicNumber(Exception):
    """Exception returned when the first 4 bytes of a data block are not the
    expected RHX TCP magic number (0x2ef07a08).
    """


if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket.
    # This is the maximum number of bytes expected for 1 read. 1024 is plenty
    # for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024

    # Declare buffer size for reading from TCP waveform socket.
    # This is the maximum number of bytes expected for 1 read.

    # There will be some TCP lag in both starting and stopping acquisition,
    # so the exact number of data blocks may vary slightly.
    # At 30 kHz with 1 channel, 1 second of wideband waveform data is
    # 181,420 byte. See 'Calculations for accurate parsing' for more details.
    # To allow for some TCP lag in stopping acquisition resulting in slightly
    # more than 1 second of data, 200000 should be a safe buffer size.
    # Increase if channels, filter bands, or acquisition time increase.
    WAVEFORM_BUFFER_SIZE = 200000

    # RHX software is hard-coded to always handle data in blocks of 128 frames.
    FRAMES_PER_BLOCK = 128

    ReadWaveformDataDemo()
