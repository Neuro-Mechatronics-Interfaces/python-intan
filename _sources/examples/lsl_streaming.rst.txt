Lab Streaming Layer (LSL) Integration
======================================

Lab Streaming Layer (LSL) is a system for synchronizing streaming data across multiple applications in real-time. These examples demonstrate how to publish Intan EMG data to LSL streams and subscribe to external LSL markers for synchronized recording.

**Requirements:**

- `pylsl` library (``pip install pylsl``)
- Intan RHX device with TCP streaming enabled
- Optional: LSL applications for marker streaming (e.g., stimulus presentation software)

----

Publishing EMG Data to LSL
----------------------------

Stream EMG data from Intan RHX device to an LSL outlet for consumption by other applications.

**Basic LSL Publisher:**

.. code-block:: python

    """
    Publish Intan EMG data to LSL stream
    """
    from intan.interface import IntanRHXDevice, LSLPublisher

    # Initialize Intan device
    device = IntanRHXDevice(num_channels=64)
    device.enable_wide_channel(range(64))
    device.start_streaming()

    # Create LSL outlet
    publisher = LSLPublisher(
        name='IntanEMG',
        stream_type='EMG',
        channel_count=64,
        sample_rate=4000,
        source_id='intan_rhx_001'
    )

    try:
        while True:
            # Get data from device
            timestamps, data = device.stream(n_frames=40)  # 10ms chunks

            # Publish to LSL
            publisher.push_chunk(data.T)  # LSL expects (samples, channels)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        publisher.close()
        device.close()

----

Subscribing to LSL Markers
----------------------------

Receive event markers from external LSL sources (e.g., task cues, stimuli).

.. code-block:: python

    """
    Subscribe to LSL marker stream
    Script: examples/LSL/lsl_marker_sub.py
    """
    from intan.interface import LSLSubscriber
    import time

    # Connect to marker stream
    subscriber = LSLSubscriber(
        stream_name='TaskMarkers',
        stream_type='Markers'
    )

    print("Waiting for markers...")
    try:
        while True:
            # Pull markers (non-blocking)
            marker, timestamp = subscriber.pull_sample(timeout=0.0)

            if marker:
                print(f"[{timestamp:.3f}] Received marker: {marker}")

            time.sleep(0.01)  # 10ms polling

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        subscriber.close()

----

LSL Waveform Viewer
--------------------

Real-time visualization of LSL streams with scrolling display.

.. code-block:: python

    """
    Real-time LSL waveform viewer
    Script: examples/LSL/lsl_waveform_viewer.py
    """
    from intan.interface import LSLSubscriber
    from intan.plotting import RealtimePlotter
    import numpy as np

    # Connect to EMG stream
    subscriber = LSLSubscriber(
        stream_type='EMG',
        resolve_timeout=5.0
    )

    print(f"Connected to: {subscriber.stream_info.name()}")
    n_channels = subscriber.stream_info.channel_count()
    fs = subscriber.stream_info.nominal_srate()

    # Initialize plotter
    plotter = RealtimePlotter(
        n_channels=n_channels,
        sample_rate=fs,
        window_sec=2.0,
        update_interval_ms=50
    )

    # Buffer for accumulating samples
    buffer_size = int(fs * 2)  # 2 second buffer
    buffer = np.zeros((n_channels, buffer_size))

    try:
        plotter.start()

        while True:
            # Pull chunk of samples
            chunk, timestamps = subscriber.pull_chunk()

            if chunk:
                chunk = np.array(chunk).T  # Shape: (channels, samples)
                n_new = chunk.shape[1]

                # Update rolling buffer
                buffer = np.roll(buffer, -n_new, axis=1)
                buffer[:, -n_new:] = chunk

                # Update plot
                plotter.update(buffer)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        plotter.stop()
        subscriber.close()

----

LSL Stacked Multi-Channel Plot
--------------------------------

Display multiple LSL channels in a stacked layout.

.. code-block:: python

    """
    Stacked plot of multiple LSL channels
    Script: examples/LSL/lsl_stacked_plot.py
    """
    from intan.interface import LSLSubscriber
    from intan.plotting import StackedPlot
    import numpy as np

    # Connect to stream
    subscriber = LSLSubscriber(stream_type='EMG')
    n_channels = subscriber.stream_info.channel_count()
    fs = subscriber.stream_info.nominal_srate()

    # Select channels to display
    channels_to_plot = range(0, min(16, n_channels))  # First 16 channels

    # Initialize stacked plot
    plot = StackedPlot(
        n_channels=len(channels_to_plot),
        channel_labels=[f"Ch {i}" for i in channels_to_plot],
        sample_rate=fs,
        window_sec=3.0,
        y_scale=200  # µV scale
    )

    buffer_size = int(fs * 3)
    buffer = np.zeros((len(channels_to_plot), buffer_size))

    try:
        plot.start()

        while True:
            chunk, _ = subscriber.pull_chunk(timeout=0.01)

            if chunk:
                chunk = np.array(chunk).T[channels_to_plot, :]
                n_new = chunk.shape[1]

                buffer = np.roll(buffer, -n_new, axis=1)
                buffer[:, -n_new:] = chunk

                plot.update(buffer)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        plot.stop()
        subscriber.close()

----

LSL RMS Bar Plot
-----------------

Display real-time RMS values across channels as animated bar chart.

.. code-block:: python

    """
    Real-time RMS bar plot from LSL stream
    Script: examples/LSL/lsl_rms_barplot.py
    """
    from intan.interface import LSLSubscriber
    from intan.processing import window_rms
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    # Connect to stream
    subscriber = LSLSubscriber(stream_type='EMG')
    n_channels = subscriber.stream_info.channel_count()
    fs = subscriber.stream_info.nominal_srate()

    # RMS calculation parameters
    RMS_WINDOW_SEC = 0.1  # 100ms RMS window
    UPDATE_INTERVAL_MS = 100

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 6))
    channel_labels = [f"Ch{i}" for i in range(n_channels)]
    x_pos = np.arange(n_channels)
    bars = ax.bar(x_pos, np.zeros(n_channels), color='cyan', edgecolor='black')

    ax.set_xlabel('Channel')
    ax.set_ylabel('RMS Amplitude (µV)')
    ax.set_title('Real-time EMG RMS Activity')
    ax.set_xticks(x_pos[::4])  # Show every 4th label
    ax.set_xticklabels(channel_labels[::4])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Buffer for RMS calculation
    window_samples = int(fs * RMS_WINDOW_SEC)
    buffer = np.zeros((n_channels, window_samples))

    def update_plot(frame):
        # Pull new data
        chunk, _ = subscriber.pull_chunk(timeout=0.0)

        if chunk:
            chunk = np.array(chunk).T  # Shape: (channels, samples)
            n_new = min(chunk.shape[1], window_samples)

            # Update buffer
            buffer[:] = np.roll(buffer, -n_new, axis=1)
            buffer[:, -n_new:] = chunk[:, -n_new:]

        # Calculate RMS
        rms_values = np.sqrt(np.mean(buffer**2, axis=1))

        # Update bars
        for bar, rms in zip(bars, rms_values):
            bar.set_height(rms)

        return bars

    # Animation
    ani = animation.FuncAnimation(
        fig, update_plot,
        interval=UPDATE_INTERVAL_MS,
        blit=True,
        cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        subscriber.close()

----

Synchronized Recording with Markers
-------------------------------------

Record EMG data while synchronizing with external event markers.

.. code-block:: python

    """
    Record EMG with synchronized markers
    """
    from intan.interface import LSLSubscriber, LSLPublisher
    from intan.interface import IntanRHXDevice
    import numpy as np
    from datetime import datetime

    # Start Intan device
    device = IntanRHXDevice(num_channels=64)
    device.enable_wide_channel(range(64))
    device.start_streaming()

    # Subscribe to marker stream
    marker_sub = LSLSubscriber(stream_type='Markers')

    # Publish EMG to LSL (optional, for other apps to use)
    emg_pub = LSLPublisher(
        name='IntanEMG',
        stream_type='EMG',
        channel_count=64,
        sample_rate=4000
    )

    # Recording buffers
    emg_buffer = []
    emg_timestamps = []
    markers = []

    print("Recording... Press Ctrl+C to stop")

    try:
        while True:
            # Get EMG data
            ts, data = device.stream(n_frames=40)
            emg_buffer.append(data)
            emg_timestamps.append(ts)

            # Publish to LSL
            emg_pub.push_chunk(data.T)

            # Check for markers
            marker, marker_ts = marker_sub.pull_sample(timeout=0.0)
            if marker:
                markers.append((marker_ts, marker))
                print(f"[{marker_ts:.3f}] Marker: {marker}")

    except KeyboardInterrupt:
        print("\\nStopping recording...")

    finally:
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        emg_data = np.concatenate(emg_buffer, axis=1)
        np.savez(
            f'recording_{timestamp}.npz',
            emg=emg_data,
            timestamps=np.concatenate(emg_timestamps),
            markers=markers,
            sample_rate=4000
        )

        print(f"Saved: recording_{timestamp}.npz")
        print(f"EMG shape: {emg_data.shape}")
        print(f"Markers received: {len(markers)}")

        # Cleanup
        emg_pub.close()
        marker_sub.close()
        device.close()

----

LSL Stream Discovery
---------------------

Find and list all available LSL streams on the network.

.. code-block:: python

    """
    Discover all available LSL streams
    """
    from pylsl import resolve_streams
    import time

    print("Searching for LSL streams...")
    streams = resolve_streams(wait_time=2.0)

    if not streams:
        print("No LSL streams found.")
    else:
        print(f"Found {len(streams)} stream(s):\\n")

        for i, stream in enumerate(streams):
            print(f"Stream {i+1}:")
            print(f"  Name: {stream.name()}")
            print(f"  Type: {stream.type()}")
            print(f"  Channels: {stream.channel_count()}")
            print(f"  Sample Rate: {stream.nominal_srate()} Hz")
            print(f"  Source: {stream.source_id()}")
            print()

----

Performance Tips
-----------------

1. **Chunk Size**: Process data in appropriately-sized chunks (10-50ms) to balance latency and efficiency

2. **Threading**: Use separate threads for LSL I/O and processing:

   .. code-block:: python

       from threading import Thread

       def lsl_reader_thread():
           while running:
               chunk, ts = subscriber.pull_chunk()
               data_queue.put((chunk, ts))

3. **Buffer Management**: Use circular buffers for efficient memory usage

4. **Time Synchronization**: LSL provides automatic clock synchronization across devices

----

See Also
---------

- :doc:`live_plotting` - Real-time plotting techniques
- :doc:`../info/advanced` - Advanced streaming concepts
- `LSL Documentation <https://labstreaminglayer.readthedocs.io/>`_
