# Hardware interface examples

These scripts demonstrate optional external hardware. Run commands from the repository root.

## RHX connection

Start the TCP server in Intan RHX software, then inspect the available options:

```bash
python examples/interface/rhx_connect_to_device.py --help
python examples/interface/rhx_connect_to_device.py --channels 32 --verbose
```

The default RHX command and waveform ports are 5000 and 5001.

## Combined RHX and Pico IMU logging

`rhx_emg_and_imu.py` imports `pico_imu_client.py` from the same directory and writes the most recent IMU sample beside each EMG sample:

```bash
python examples/interface/rhx_emg_and_imu.py --help
python examples/interface/rhx_emg_and_imu.py --channels 32 --outfile emg_imu.csv
```

This needs both an RHX TCP stream and compatible Pico firmware. If Pico discovery fails, the script continues with `NaN` IMU columns.

## Pico client and firmware

```bash
python examples/interface/pico_imu_client.py --help
```

The host client uses UDP discovery. Firmware source is in `examples/interface/pico/code.py` and must be deployed with its CircuitPython dependencies. It is not desktop Python and is not imported by the package.

## Limitations

- No hardware integration test can run without the corresponding device and firmware.
- Firewalls may block UDP discovery or the RHX TCP ports.
- Stop acquisition with Ctrl+C so streams and output files close cleanly.
