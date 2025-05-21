"""
intan.io._exceptions

Custom exception classes used throughout the `intan` package.

These exceptions provide clearer debugging information for:
- Invalid or corrupted `.rhd` files
- Mismatched channel definitions
- Broken TCP streams from the RHX server
- File size or format inconsistencies

Each exception inherits from `Exception` and includes a brief description.
"""

class UnrecognizedFileError(Exception):
    """Exception returned when reading a file as an RHD header yields an
    invalid magic number (indicating this is not an RHD header file).
    """


class UnknownChannelTypeError(Exception):
    """Exception returned when a channel field in RHD header does not have
    a recognized signal_type value. Accepted values are:
    0: amplifier channel
    1: aux input channel
    2: supply voltage channel
    3: board adc channel
    4: dig in channel
    5: dig out channel
    """


class FileSizeError(Exception):
    """Exception returned when file reading fails due to the file size
    being invalid or the calculated file size differing from the actual
    file size.
    """


class QStringError(Exception):
    """Exception returned when reading a QString fails because it is too long.
    """


class ChannelNotFoundError(Exception):
    """Exception returned when plotting fails due to the specified channel
    not being found.
    """


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
