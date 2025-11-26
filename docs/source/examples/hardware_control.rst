Hardware Integration and Control
==================================

These examples demonstrate integration between Intan EMG systems and external hardware for closed-loop control, robotic applications, and multi-sensor systems.

----

3D-Printed Robotic Arm Control
--------------------------------

Control a 3D-printed robotic hand using EMG-based gesture recognition. This example shows how to integrate the Intan system with a Raspberry Pi Pico-controlled servo system.

.. image:: ../../assets/arm-moving.gif
    :alt: 3D-printed arm in action
    :width: 600
    :align: center

**Hardware Requirements:**

- Raspberry Pi Pico (or Pico 2)
- PCA9685 16-channel servo driver
- 5-6 servo motors
- 3D-printed InMoov hand components
- 5V power supply (servos)

**Software Requirements:**

- CircuitPython on Pico
- Python 3.10+ on host PC
- PySerial library

**Setup:**

See the detailed `build guide <https://github.com/Neuro-Mechatronics-Interfaces/python-intan/tree/main/examples/3D_printed_arm_control>`_.

**Architecture:**

::

    ┌─────────────┐         ┌──────────────┐         ┌─────────────┐
    │ Intan RHX   │  USB    │   Host PC    │  Serial │ Pico + PCA  │
    │   Device    │────────▶│  + Python    │────────▶│   9685      │
    └─────────────┘         └──────────────┘         └─────────────┘
                                   │                         │
                                   │ EMG Processing          │ Servo Control
                                   │ Gesture Recognition     │
                                   ▼                         ▼
                            [Predict Gesture]          [Move Fingers]

**Host-side control script:**

.. code-block:: python

    """
    EMG-controlled robotic hand
    Requires: Pico running code.py from examples/3D_printed_arm_control/
    """
    from intan.interface import IntanRHXDevice
    from intan.ml import EMGRealTimePredictor
    import serial
    import tensorflow as tf

    # Initialize Intan device
    device = IntanRHXDevice(num_channels=64)
    device.enable_wide_channel(range(64))
    device.start_streaming()

    # Load gesture model
    model = tf.keras.models.load_model('gesture_model.keras')

    # Connect to Pico via serial
    pico = serial.Serial('COM3', 115200, timeout=1)  # Adjust port

    # Gesture-to-command mapping
    GESTURE_COMMANDS = {
        'rest': 'rest',
        'flex': 'grip',
        'extend': 'open',
        'pinch': 'pinch',
        'point': 'point',
    }

    # Real-time predictor
    predictor = EMGRealTimePredictor(
        device=device,
        model=model,
        pca=pca,
        mean=mean,
        std=std,
        label_names=list(GESTURE_COMMANDS.keys()),
        confidence_threshold=0.8
    )

    print("System ready. Perform gestures to control robot.")

    try:
        predictor.run_prediction_loop()

        while True:
            prediction = predictor.get_prediction()

            if prediction:
                gesture = prediction['label']
                confidence = prediction['confidence']

                if gesture in GESTURE_COMMANDS:
                    command = GESTURE_COMMANDS[gesture]
                    pico.write(f"{command}\\n".encode())
                    print(f"[{confidence:.1%}] {gesture} → {command}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pico.write(b"rest\\n")
        pico.close()
        device.close()

**Pico firmware (CircuitPython):**

.. code-block:: python

    """
    Raspberry Pi Pico servo controller
    File: examples/3D_printed_arm_control/code.py
    """
    import board
    import busio
    from adafruit_servokit import ServoKit
    import supervisor

    # Initialize PCA9685
    i2c = busio.I2C(board.GP1, board.GP0)
    kit = ServoKit(channels=16, i2c=i2c)

    # Servo channel assignments
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4
    WRIST = 5

    # Define gestures as servo angles
    GESTURES = {
        'rest': [90, 90, 90, 90, 90, 90],
        'grip': [170, 170, 170, 170, 170, 90],
        'open': [10, 10, 10, 10, 10, 90],
        'pinch': [170, 170, 10, 10, 10, 90],
        'point': [170, 10, 170, 170, 170, 90],
    }

    def move_to_gesture(gesture_name):
        if gesture_name in GESTURES:
            angles = GESTURES[gesture_name]
            for servo_idx, angle in enumerate(angles):
                kit.servo[servo_idx].angle = angle

    # Main loop
    print("Robotic hand ready")
    while True:
        if supervisor.runtime.serial_bytes_available:
            command = input().strip()
            print(f"Received: {command}")
            move_to_gesture(command)

**Available gestures:**

- ``rest``: Neutral position
- ``grip``: Close all fingers
- ``open``: Open all fingers
- ``pinch``: Thumb + index finger
- ``point``: Extend index finger
- ``flex``: Individual finger control
- ``supinate/pronate``: Wrist rotation

----

RHX Device Interface Examples
-------------------------------

Low-level examples for interfacing with Intan RHX hardware.

**Connecting to device:**

.. code-block:: python

    """
    Basic device connection and configuration
    Script: examples/interface/rhx_connect_to_device.py
    """
    from intan.interface import IntanRHXDevice

    # Connect to RHX TCP server
    device = IntanRHXDevice(
        host='127.0.0.1',
        command_port=5000,
        waveform_port=5001
    )

    print(f"Connected: {device.connected}")
    print(f"Sample rate: {device.get_sample_rate()} Hz")
    print(f"Available channels: {device.get_num_channels()}")

    # Enable specific channels
    device.enable_wide_channel(range(0, 64))

    # Start streaming
    device.start_streaming()

    # Stream 1 second of data
    timestamps, data = device.stream(duration_sec=1.0)
    print(f"Streamed data shape: {data.shape}")

    device.close()

**Multi-channel streaming plot:**

.. code-block:: bash

    # Real-time plot of multiple channels
    python examples/RHXDevice/multichannel_stream_plot.py

**Recording demo:**

.. code-block:: bash

    # Record streaming data to file
    python examples/RHXDevice/record_demo.py --duration 60 --output recording.npz

**TCP benchmark:**

.. code-block:: bash

    # Test TCP streaming performance
    python examples/RHXDevice/tcp_benchmark.py --channels 128 --duration 10

**Comparing file vs stream:**

.. code-block:: python

    """
    Verify streaming data matches recorded file
    Script: examples/RHXDevice/compare_rhd_vs_stream.py
    """
    from intan.io import load_rhd_file
    from intan.interface import IntanRHXDevice
    import numpy as np

    # Stream live data
    device = IntanRHXDevice()
    device.enable_wide_channel([15])  # Channel 15
    device.start_streaming()
    _, stream_data = device.stream(duration_sec=2.0)
    device.close()

    # Load same data from file
    result = load_rhd_file('recording.rhd')
    file_data = result['amplifier_data'][15, :len(stream_data[0])]

    # Compare
    correlation = np.corrcoef(stream_data[0], file_data)[0, 1]
    print(f"Correlation: {correlation:.4f}")

    # Should be >0.99 if synchronized properly

**Scrolling live plot:**

.. code-block:: python

    """
    Real-time scrolling plot
    Script: examples/RHXDevice/scrolling_live.py
    """
    from intan.interface import IntanRHXDevice
    from intan.plotting import RealtimePlotter
    import numpy as np

    device = IntanRHXDevice(num_channels=8)
    device.enable_wide_channel(range(8))
    device.start_streaming()

    plotter = RealtimePlotter(
        n_channels=8,
        sample_rate=4000,
        window_sec=2.0,
        update_interval_ms=50
    )

    try:
        plotter.start()
        buffer_size = 8000
        buffer = np.zeros((8, buffer_size))

        while True:
            _, data = device.stream(n_frames=200)

            buffer = np.roll(buffer, -200, axis=1)
            buffer[:, -200:] = data

            plotter.update(buffer)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        plotter.stop()
        device.close()

----

Raspberry Pi Pico + IMU Integration
-------------------------------------

Combine Intan EMG with IMU (accelerometer/gyroscope) data from a Pico.

.. code-block:: python

    """
    Synchronized EMG + IMU recording
    Script: examples/interface/rhx_emg_and_imu.py
    """
    from intan.interface import IntanRHXDevice, PicoIMUClient
    import numpy as np

    # Connect to Intan
    emg_device = IntanRHXDevice(num_channels=64)
    emg_device.enable_wide_channel(range(64))
    emg_device.start_streaming()

    # Connect to Pico IMU
    imu_client = PicoIMUClient(port='COM4', baud_rate=115200)

    # Recording buffers
    emg_buffer = []
    imu_buffer = []
    timestamps = []

    print("Recording EMG + IMU... Press Ctrl+C to stop")

    try:
        while True:
            # Get EMG data
            ts, emg_data = emg_device.stream(n_frames=40)
            emg_buffer.append(emg_data)

            # Get IMU data (non-blocking)
            imu_sample = imu_client.read_sample()
            if imu_sample:
                imu_buffer.append(imu_sample)

            timestamps.append(ts)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        # Save synchronized data
        np.savez(
            'emg_imu_recording.npz',
            emg=np.concatenate(emg_buffer, axis=1),
            imu=np.array(imu_buffer),
            timestamps=np.concatenate(timestamps)
        )

        print(f"Saved EMG shape: {np.concatenate(emg_buffer, axis=1).shape}")
        print(f"Saved IMU samples: {len(imu_buffer)}")

        imu_client.close()
        emg_device.close()

**Pico IMU streaming script:**

.. code-block:: python

    """
    Pico IMU data publisher
    File: examples/interface/pico/code.py
    """
    import board
    import busio
    import adafruit_mpu6050
    import time

    i2c = busio.I2C(board.GP1, board.GP0)
    mpu = adafruit_mpu6050.MPU6050(i2c)

    while True:
        accel = mpu.acceleration
        gyro = mpu.gyro

        # Send as CSV over serial
        print(f"{time.monotonic()},{accel[0]},{accel[1]},{accel[2]},"
              f"{gyro[0]},{gyro[1]},{gyro[2]}")

        time.sleep(0.01)  # 100 Hz

----

Wireless EMG with RHD2164
--------------------------

Interface with RHD2164 wireless headstage via SPI.

.. code-block:: python

    """
    Wireless RHD2164 interface
    Script: examples/RHD2164_wireless/code.py
    """
    from intan.interface import RHD2164Interface
    import time

    # Initialize SPI interface (on Pico or similar)
    rhd = RHD2164Interface(
        spi_bus=0,
        cs_pin=5,
        sample_rate=20000
    )

    # Configure channels
    rhd.configure_channels(
        channels=range(64),
        bandwidth=[0.1, 7500],  # Hz
        dsp_enabled=True
    )

    # Start acquisition
    rhd.start()

    try:
        while True:
            if rhd.data_available():
                samples = rhd.read_samples(100)  # Read 100 samples
                # Process samples...
                print(f"Read {len(samples)} samples")

            time.sleep(0.01)

    except KeyboardInterrupt:
        rhd.stop()

----

TCP Command Interface
----------------------

Low-level TCP control examples.

.. code-block:: python

    """
    Send commands to RHX via TCP
    Script: examples/intan_tcp/RHXRunAndStimulateDemo.py
    """
    from intan.interface import IntanTCPClient

    client = IntanTCPClient()

    # Query device info
    sample_rate = client.execute_command('get sampleratehertz')
    print(f"Sample rate: {sample_rate} Hz")

    # Configure stimulation
    client.execute_command('set a-010.stimenabled true')
    client.execute_command('set a-010.stimshape biphasic')
    client.execute_command('set a-010.firstphaseamplitudemicroamps 100')
    client.execute_command('set a-010.firstphasedurationmicroseconds 200')

    # Trigger stimulation
    client.execute_command('execute manualstimtriggerpulse a-010')

    client.close()

----

See Also
---------

- :doc:`gesture_classification` - ML pipeline for control
- :doc:`../info/rhx_device` - RHX device details
- `InMoov hand build guide <https://inmoov.fr/hand-and-forarm/>`_
- `CircuitPython documentation <https://circuitpython.org/>`_
