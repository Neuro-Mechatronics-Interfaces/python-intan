Intan RHX Controller Device
========================================================

This tutorial demonstrates how to connect to an Intan RHX system using the ``IntanRHXDevice`` class and interact with it via the Remote TCP Control interface.

.. important::

   Before starting, **make sure** the Remote TCP Control GUI in the Intan RHX software has both servers enabled (Command and Data ports).
   The default ports are 5000 (Command) and 5001 (Data).

----

Connect to an Intan RHX Controller
---------------------------------------------

The following example show how simple it is to connect to an RHX device:

.. code-block:: python

    from intan.rhx_interface import IntanRHXDevice

    SAMPLE_RATE = 4000.0      # Specify sampling rate in Hz, ignored if using 'playback' mode
    NUM_CHANNELS = 32         # Number of channels to read from the intan device

    # Initialize device
    device = IntanRHXDevice(sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)

----

We can enable streaming for all channels:

.. code-block:: python

   # Enable "wide" channel streaming for all channels (or indices for specific channels)
    device.enable_wide_channel(range(NUM_CHANNELS))

----

Record EMG Data Duration
------------------------------------------------

Let's records a 10-second EMG segment and plots the result:

.. code-block:: python

    from intan.plotting import plot_figure
    import numpy as np

    DURATION_SEC = 10

    # Record EMG for the specified duration (seconds)
    emg = device.record(duration_sec=DURATION_SEC)

    # Safely stop the device and close sockets
    device.set_run_mode("stop")

    # Plot the EMG data from channel 6 (index 5)
    if emg is not None:
        t = np.arange(emg.shape[1]) / SAMPLE_RATE
        plot_figure(emg[5], t, title="EMG Signal", x_label="Time (s)", y_label="Amplitude (μV)", legend=False)

----

Remember to clode the device connection when done:

.. code-block:: python

    # Close the device connection
    device.close()

----


Configuring Channels and Streaming Options
------------------------------------------------

The ``IntanRHXDevice`` inherits configuration commands, so you can enable or disable specific channel types for streaming over TCP:

.. code-block:: python

    # Enable wideband streaming for channels 0-15
    device.enable_wide_channel(range(16), status=True)

    # Enable highpass streaming for channels 0-7
    device.enable_high_channel(range(8), status=True)

    # Disable all data outputs (recommended before changing channel configs)
    device.clear_all_data_outputs()

----

Changing Sample Rate and Device Parameters
------------------------------------------------

You can read or set core device parameters programmatically:

.. code-block:: python

    # Get current sample rate (Hz)
    fs = device.get_sample_rate()

    # Set sample rate to 5000 Hz (if supported)
    device.set_sample_rate(5000)

    # Set the TCP write block size (number of data blocks sent per packet)
    device.set_blocks_per_write(2)

----

Using Context Management for Automatic Cleanup
------------------------------------------------

The ``IntanRHXDevice`` supports Python's ``with`` statement for safe usage:

.. code-block:: python

    from intan.rhx_interface import IntanRHXDevice

    with IntanRHXDevice(sample_rate=4000, num_channels=128) as device:
        device.enable_wide_channel(range(128))
        emg = device.record(duration_sec=5)
        # Device is safely stopped and closed automatically

----

Troubleshooting Connection Issues
------------------------------------------------

- **"Connection refused" or "timeout":**
  Ensure the Intan RHX software's Remote TCP Control servers are running (both Command and Data).
  Double-check the host address and port numbers.

- **Device not streaming or empty data:**
  Confirm the correct channel types are enabled (`enable_wide_channel`, etc.) and data output is enabled for the channels you want.

- **Recording length mismatch:**
  Verify the sample rate matches your hardware/software setting in RHX.

----

More: Streaming and Real-Time Visualization
------------------------------------------------

For live scrolling plots or real-time processing, see the ``multichannel_stream_plot.py`` and ``scrolling_live.py`` examples in the examples folder, or refer to the **Streaming** tutorial.

----

Summary
------------------------------------------------

You can fully automate connection, configuration, and recording from the Intan RHX device using the ``IntanRHXDevice`` class.
For advanced options (triggered acquisition, analog or spike channel streaming, or disk recording), refer to the API Reference and the configuration commands available in the ``RHXConfig`` base class.

