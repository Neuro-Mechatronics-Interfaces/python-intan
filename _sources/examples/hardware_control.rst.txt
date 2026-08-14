Hardware integration
====================

The hardware examples connect Intan acquisition workflows to optional external
devices. They are reference integrations, not turnkey applications: no trained
models, hardware drivers, or device-specific credentials are bundled.

3D-printed arm control
----------------------

``examples/3D_printed_arm_control`` contains CircuitPython firmware for a
Raspberry Pi Pico controlling a PCA9685 servo driver. Typical hardware includes
a Pico or Pico 2, five or six servos, a suitable 5 V servo supply, and an InMoov
or comparable printed hand.

Use the supported PyTorch workflow in ``examples/gesture_classifier`` to build
and run a gesture classifier. The application consuming its predictions should
map labels such as ``rest``, ``grip``, and ``open`` to the newline-terminated
commands accepted by the Pico firmware. Keep servo power separate from USB
power and test motion limits before connecting the printed mechanism.

Pico setup
~~~~~~~~~~

1. Install a compatible CircuitPython release on the Pico.
2. Install the PCA9685 CircuitPython library on the device.
3. Copy ``examples/3D_printed_arm_control/code.py`` to the Pico as ``code.py``.
4. Confirm the serial port and servo channel mapping before sending commands.

The pinned firmware image stored in the repository is retained for hardware
reproducibility but is excluded from Python release archives.

RHX TCP examples
----------------

``examples/RHXDevice`` and ``examples/interface`` demonstrate acquisition from
an RHX TCP server. Start RHX, enable its TCP server, and confirm the configured
command and waveform ports before running these scripts. Use ``--help`` for
scripts that expose command-line options.

Pico IMU example
----------------

``examples/interface/pico_imu_client.py`` receives optional IMU data from the
matching CircuitPython firmware in ``examples/interface/pico/code.py``. This is
a separate serial integration and is not required for normal RHD file loading.

Safety and limitations
----------------------

- Test hardware motion without a person attached.
- Use a current-limited supply sized for the servos.
- Treat serial port names, channel counts, and sampling rates as deployment
  configuration rather than library defaults.
- Hardware examples require their physical devices and were not exercised by
  the package's headless test suite.
