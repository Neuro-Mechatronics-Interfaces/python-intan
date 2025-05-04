# Auto-generated from _loaders.py

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

