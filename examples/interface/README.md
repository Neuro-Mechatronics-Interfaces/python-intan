# Interface Examples

Hardware communication and device control examples.

## Examples

### rhx_connect_to_device.py

Connect to Intan RHX software and stream EMG data.

```bash
# Basic connection (no LSL output)
python rhx_connect_to_device.py --channels 128 --verbose

# With LSL streaming (makes data available to other applications)
python rhx_connect_to_device.py \
    --channels 128 \
    --use_lsl \
    --lsl_numeric_name "EMG" \
    --lsl_numeric_type "EMG" \
    --verbose

# Custom network settings
python rhx_connect_to_device.py \
    --host 192.168.1.100 \
    --command_port 5000 \
    --data_port 5001 \
    --channels 64 \
    --channel_port b
```

### rhx_emg_and_imu.py

Stream both EMG and IMU data simultaneously.

```bash
python rhx_emg_and_imu.py --emg_channels 0:64 --imu_source pico
```

### pico_imu_client.py

Connect to a Raspberry Pi Pico running IMU firmware.

```bash
python pico_imu_client.py --port /dev/ttyACM0 --baud 115200
```

## Pico Firmware

The `pico/` subdirectory contains CircuitPython firmware for Raspberry Pi Pico:
- IMU streaming (accelerometer, gyroscope)
- Serial communication with the host PC
- Configurable sample rates

See `pico/README.md` for installation instructions.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Host Computer                            │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │ rhx_connect_    │    │ LSLPublisher    │    │ LSL Viewers  │  │
│  │ to_device.py    │───▶│ (optional)      │───▶│ (other apps) │  │
│  └────────┬────────┘    └─────────────────┘    └──────────────┘  │
│           │                                                      │
└───────────┼──────────────────────────────────────────────────────┘
            │ TCP/IP
            ▼
┌───────────────────────┐
│  Intan RHX Software   │
│  (TCP Server)         │
│  Ports: 5000, 5001    │
└───────────┬───────────┘
            │ USB
            ▼
┌───────────────────────┐
│  Intan Headstage      │
│  (RHD2164, etc.)      │
└───────────────────────┘
```

## Command Reference

### IntanRHXDevice Methods

```python
from intan.interface import IntanRHXDevice, LSLOptions

# Initialize device connection
device = IntanRHXDevice(
    host="127.0.0.1",
    command_port=5000,
    data_port=5001,
    num_channels=128,
    sample_rate=None,         # Auto-detect from RHX
    buffer_duration_sec=5.0,
    auto_start=False,
    use_lsl=False,
    lsl_options=LSLOptions(...),
    verbose=True,
)

# Channel control
device.clear_all_data_outputs()           # Disable all outputs
device.enable_wide_channel(0, port='a')   # Enable single channel
device.enable_wide_channel(range(64), port='a')  # Enable range

# Streaming control
device.start_streaming()                  # Begin data acquisition
device.stop_streaming()                   # Pause acquisition

# Data access
window = device.get_latest_window(200)    # Last 200ms, shape (C, N)
device.read_window(seconds=1.0)           # Last 1 second

# Cleanup
device.close()
```

### LSLOptions Configuration

```python
from intan.interface import LSLOptions

lsl_opts = LSLOptions(
    numeric_name="EMG",           # Stream name for numeric data
    numeric_type="EMG",           # Stream type
    with_markers=False,           # Include marker stream?
    chunk_size=32,                # Samples per LSL push
    source_id="IntanRHX_001",     # Unique source identifier
)
```

## Troubleshooting

### "Failed to connect to RHX device"

1. Ensure RHX software is running
2. Enable TCP server in RHX: Network → TCP → Enable Server
3. Check firewall settings for ports 5000/5001
4. Verify IP address if connecting remotely

### "No data received"

1. Check that channels are enabled in RHX
2. Verify RHX is in "Run" mode (not "Stop")
3. Ensure sample rate settings match

### LSL stream not visible

1. Verify `--use_lsl` flag is set
2. Check LSL library is installed: `pip install pylsl`
3. Look for the stream: `python -c "from pylsl import resolve_streams; print(resolve_streams())"`
