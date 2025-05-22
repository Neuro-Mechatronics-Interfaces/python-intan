"""
intan.rhx_interface._rhx_config

Low-level TCP configuration interface for the Intan RHX system.

This class wraps `set` and `get` TCP commands to configure:
- Analog channel streaming (wide, highpass, lowpass, spike)
- Sample rate and block size
- Digital inputs and trigger behavior
- Disk recording parameters and filename handling
- Filter settings (DSP, notch, bandwidth)

This module is inherited by `IntanRHXDevice`, which handles data streaming.
"""

import time
from collections.abc import Iterable
from intan.io import GetSampleRateFailure


class RHXConfig:
    """
    Stream and record EMG data from the Intan RHX system.

    Inherits:
        RHXConfig: Command/control interface for RHX configuration

    Responsibilities:
        - Establish TCP connections to RHX server
        - Parse incoming EMG waveform blocks
        - Buffer, store, or visualize real-time EMG data
        - Record sessions of configurable duration

    Attributes:
        host (str): IP address of the RHX system.
        command_port (int): Port for command/control TCP socket.
        data_port (int): Port for binary waveform data.
        num_channels (int): Number of EMG channels enabled.
        sample_rate (float): Sampling rate in Hz.
        verbose (bool): Debug logging toggle.
    """
    def __init__(self, command_socket, send_delay=0.05, verbose=False):
        self.command_socket = command_socket
        self.send_delay = send_delay
        self.verbose = verbose

    # === Core Functions ===
    def set_parameter(self, param, value):
        """
        Set a parameter on the RHX system.

        Parameters:
            param (str): Parameter to set.
            value (str): Value to set the parameter to.
            send_delay (float): Delay after sending the command.

        Raises:
            ValueError: If the parameter is not recognized.
        """
        self.command_socket.sendall(f"set {param} {value}\n".encode())
        time.sleep(self.send_delay)

    def get_parameter(self, param):
        """
        Get a parameter from the RHX system.

        Parameters:
            param (str): Parameter to get.

        Returns:
            str: Value of the parameter.
        """
        msg = f"get {param}"
        self.command_socket.sendall(msg.encode())
        return self.command_socket.recv(1024).decode()

    # === Data Output Options ===

    def enable_wide_channel(self, channels, status=True):
        """
        Enable or disable wide channel data output.

        Parameters:
            channels (int, range, or iterable): Channel numbers to enable/disable.
            status (bool): True to enable, False to disable.

        Raises:
            TypeError: If channels is not an int, range, or iterable.
        """
        if isinstance(channels, int):
            channels = [channels]
        elif not isinstance(channels, Iterable):
            raise TypeError("Channels must be an int, range, or iterable list.")

        self.clear_all_data_outputs()
        for ch in channels:
            name = f"a-{ch:03d}"
            self.set_parameter(f"{name}.tcpdataoutputenabled", 'true' if status else 'false')

    def enable_high_channel(self, channels, status=True):
        """
        Enable or disable high channel data output.

        Parameters:
            channels (int, range, or iterable): Channel numbers to enable/disable.
            status (bool): True to enable, False to disable.

        Raises:
            TypeError: If channels is not an int, range, or iterable.
        """
        if isinstance(channels, int):
            channels = [channels]
        elif not isinstance(channels, Iterable):
            raise TypeError("Channels must be an int, range, or iterable list.")

        self.clear_all_data_outputs()
        for ch in channels:
            name = f"a-{ch:03d}"
            self.set_parameter(f"{name}.tcpdataoutputenabledhigh", 'true' if status else 'false')

    def enable_low_channel(self, channels, status=True):
        """
        Enable or disable low channel data output.

        Parameters:
            channels (int, range, or iterable): Channel numbers to enable/disable.
            status (bool): True to enable, False to disable.

        Raises:
            TypeError: If channels is not an int, range, or iterable.
        """
        if isinstance(channels, int):
            channels = [channels]
        elif not isinstance(channels, Iterable):
            raise TypeError("Channels must be an int, range, or iterable list.")

        self.clear_all_data_outputs()
        for ch in channels:
            name = f"a-{ch:03d}"
            self.set_parameter(f"{name}.tcpdataoutputenabledlow", 'true' if status else 'false')

    def enable_spike_channel(self, channels, status=True):
        """
        Enable or disable spike channel data output.

        Parameters:
            channels (int, range, or iterable): Channel numbers to enable/disable.
            status (bool): True to enable, False to disable.

        Raises:
            TypeError: If channels is not an int, range, or iterable.
        """
        if isinstance(channels, int):
            channels = [channels]
        elif not isinstance(channels, Iterable):
            raise TypeError("Channels must be an int, range, or iterable list.")

        self.clear_all_data_outputs()
        for ch in channels:
            name = f"a-{ch:03d}"
            self.set_parameter(f"{name}.tcpdataoutputenabledspike", 'true' if status else 'false')

    def clear_all_data_outputs(self):
        """
        Clear all data output settings.
        """
        self.command_socket.sendall(b"execute clearalldataoutputs\n")
        time.sleep(self.send_delay)

    def get_run_mode(self):
        """
        Get the current run mode of the RHX system.

        Returns:
            str: Current run mode, either "run" or "stop".
        """
        response = self.get_parameter("runmode")
        return response.strip().split()[-1]

    def set_run_mode(self, mode):
        """
        Set the run mode of the RHX system.

        Parameters:
            mode (str): Run mode to set, either "run" or "stop".

        Raises:
            AssertionError: If the mode is not "run" or "stop".
        """
        assert mode in ["run", "stop"], "Mode must be 'run' or 'stop'"
        self.set_parameter("runmode", mode)

    def get_sample_rate(self):
        """
        Get the sample rate of the RHX system.

        Returns:
            float: Sample rate in Hz.
        Raises:
            GetSampleRateFailure: If unable to get sample rate.
        """
        # Look for "Return: SampleRateHertz N" where N is the sample rate.
        resp = self.get_parameter("sampleratehertz")
        expected_return_string = "Return: SampleRateHertz "
        if resp.find(expected_return_string) == -1:
            raise GetSampleRateFailure(
                'Unable to get sample rate from server.'
            )
        # Extract the sample rate value
        if expected_return_string in resp:
            sample_rate = float(resp[len(expected_return_string):])
        else:
            raise ValueError(f"Unable to get sample rate from server: {resp}")

        return sample_rate

    def set_sample_rate(self, rate_hz: int):
        """
        Set the sample rate of the RHX system.

        Parameters:
            rate_hz (int): Sample rate in Hz.

        Raises:
            ValueError: If the sample rate cannot be changed or is mismatched.
        """
        self.set_parameter("sampleratehertz", rate_hz)
        resp = self.get_parameter("sampleratehertz").strip()

        # Check if the rate is already set
        if "SampleRateHertz cannot be changed" in resp:
            if f"SampleRateHertz {rate_hz}" in resp:
                if self.verbose:
                    print(f"[CONFIG] Sample rate already set to {rate_hz} Hz")
                return
            else:
                raise ValueError(f"Sample rate mismatch: {resp}")

        if self.verbose:
            print(f"[CONFIG] Set sample rate response: {resp}")

    # === Save Options ===

    def set_file_format(self, fmt):
        """
        Set the file format for saving data.

        Parameters:
            fmt (str): File format to set. Options include 'intan', 'matlab', 'binary', etc.
        """
        self.set_parameter("fileformat", fmt)

    def set_filename_base(self, name):
        """
        Set the base filename for saving data.

        Parameters:
            name (str): Base filename to set.
        """
        self.set_parameter("filename.basefilename", name)

    def set_filename_path(self, path):
        """
        Set the path for saving data files.

        Parameters:
            path (str): Path to set.
        """
        self.set_parameter("filename.path", path)

    def set_create_new_directory(self, enable=True):
        """
        Set whether to create a new directory for saving data files.

        Parameters:
            enable (bool): True to create a new directory, False otherwise.
        """
        self.set_parameter("createnewdirectory", str(enable).lower())

    def set_write_to_disk_latency(self, level):
        """
        Set the write-to-disk latency level.

        Parameters:
            level (str): Latency level to set. Options include 'low', 'medium', 'high'.
        """
        self.set_parameter("writetodisklatency", level)

    # === Triggering ===

    def set_trigger_source(self, src):
        self.set_parameter("triggersource", src)

    def set_trigger_polarity(self, polarity):
        self.set_parameter("triggerpolarity", polarity)

    def set_pre_trigger_buffer(self, seconds):
        self.set_parameter("pretriggerbufferseconds", seconds)

    def set_post_trigger_buffer(self, seconds):
        self.set_parameter("posttriggerbufferseconds", seconds)

    # === Filtering ===

    def set_dsp_enabled(self, enable=True):
        self.set_parameter("dspenabled", str(enable).lower())

    def set_notch_filter(self, freq):
        self.set_parameter("notchfilterfreqhertz", freq if freq else "None")

    def set_desired_dsp_cutoff(self, freq_hz):
        self.set_parameter("desireddspcutofffreqhertz", freq_hz)

    def set_desired_lower_bandwidth(self, freq_hz):
        self.set_parameter("desiredlowerbandwidthhertz", freq_hz)

    def set_desired_upper_bandwidth(self, freq_hz):
        self.set_parameter("desiredupperbandwidthhertz", freq_hz)

    # === Digital Inputs ===
    def set_digital_input_enabled(self, dig_input=1, enable=True):
        self.set_parameter(f"DIGITAL-IN-{dig_input}.enabled", str(enable).lower())
        self.set_parameter(f"DIGITAL-IN-{dig_input}.recordingenabled", str(enable).lower())

    # === Block Settings ===
    def set_blocks_per_write(self, num_blocks):
        """
        Set the number of blocks to write to the TCP client

        Parameters:
            num_blocks (int): NUmber of blocks to write
        """
        self.set_parameter("TCPNumberDataBlocksPerWrite", num_blocks)