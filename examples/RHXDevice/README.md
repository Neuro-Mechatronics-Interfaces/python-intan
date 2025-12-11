# Intan RHX Device Examples

Direct TCP streaming from Intan RHX software, bypassing LSL.

## Prerequisites

1. **Intan RHX software** running with TCP server enabled:
   - Network → TCP → Enable TCP Data Output Server
   - Default ports: 5000 (command), 5001 (data)

2. **Intan hardware connected** (RHD2000 or RHS2000 series)

## Examples

### multichannel_stream_plot.py

Real-time waveform visualization using matplotlib animation.

```bash
python multichannel_stream_plot.py
```

**Features:**
- Displays 6 channels simultaneously (indices 10-15)
- Auto-scales to ±200 µV
- Updates at 10 Hz for smooth animation
- Uses context manager for proper cleanup

**Configuration** (edit the script):
```python
CHANNELS = [10, 11, 12, 13, 14, 15]  # Channel indices to stream
WINDOW_MS = 200                       # Display window in milliseconds
UPDATE_INTERVAL_MS = 100              # Matplotlib animation interval
```

### scrolling_live.py

Scrolling waveform display with PyQtGraph (faster than matplotlib).

```bash
python scrolling_live.py
```

**Features:**
- Displays first 8 channels by default (edit `NUM_CHANNELS` in script)
- Scrolling display with 2-second window
- Real-time autoscaling
- More efficient than matplotlib for high-rate updates

### record_demo.py

Basic recording example with single-channel visualization.

```bash
python record_demo.py
```

**Features:**
- Records 10 seconds from channels 10-15
- Displays channel 10 waveform after recording
- Saves data to `emg_data.npz` (optional - see script for enable)
- Shows proper context manager usage pattern

### tcp_benchmark.py

Low-level TCP throughput measurement (no data parsing).

```bash
python tcp_benchmark.py --sec 20 --channels 0-127
```

**Options:**
- `--sec DURATION`: Benchmark duration in seconds (default: 20.0)
- `--channels SPEC`: Channel range/list (e.g. `0-127` or `15,16,17`)
- `--blocks N`: Set `TCPNumberDataBlocksPerWrite` (optional)
- `--read_bytes N`: recv() buffer size (default: 262144)

**Output:**
```
[BENCH] Sample rate: 20000.0 Hz
[BENCH] Active channels: 128
[SUMMARY] Benchmark Results
Duration:        20.00s
Total received:  87.891 MB
Average rate:    4.395 MB/s (~22272 frames/s)
Theoretical:     ~4.883 MB/s (~20000 frames/s)
Efficiency:      90.0%
```

### compare_rhd_vs_stream.py

Validate TCP stream matches recorded RHD file (for debugging).

```bash
python compare_rhd_vs_stream.py path/to/recording.rhd
```

**Features:**
- Streams live data and compares to saved recording
- Plots difference between file and stream
- Reports statistics (mean/std/max difference)
- Useful for verifying TCP protocol implementation

## Connection Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Intan RHX Software                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────┐  │
│  │  RHD    │  │ Signal  │  │  TCP Server                 │  │
│  │ Headstage├─▶ Chain   ├─▶│  Port 5000: Commands        │  │
│  └─────────┘  └─────────┘  │  Port 5001: Data Stream     │  │
│                            └──────────────┬──────────────┘  │
└───────────────────────────────────────────┼─────────────────┘
                                            │ TCP/IP
                                            ▼
                              ┌─────────────────────────────┐
                              │    IntanRHXDevice           │
                              │    (Python client)          │
                              │                             │
                              │  • Command socket (5000)    │
                              │  • Data socket (5001)       │
                              │  • Circular buffer          │
                              │  • Optional LSL output      │
                              └─────────────────────────────┘
```

## RHX TCP Protocol

### Command Port (5000)

Send text commands, receive responses:
```python
device.send_command("set a-000.enabled true")
response = device.send_command("get a-000.enabled")
```

Common commands:
- `set runmode run` / `set runmode stop`
- `set <channel>.enabled true/false`
- `get sampleratehertz`
- `execute clearalldataoutputs`

### Data Port (5001)

Binary stream of wideband samples. Format depends on configuration:
- Each data block contains `FRAMES_PER_BLOCK` samples per channel
- Data is interleaved by channel
- Values are signed 16-bit integers (convert to µV with gain)

## Troubleshooting

### "Could not connect to RHX TCP server"

1. Verify RHX is running
2. Check TCP server is enabled in RHX settings
3. Confirm ports 5000/5001 are not blocked by firewall
4. Try connecting from localhost first

### No data received

1. Ensure channels are enabled in RHX
2. Check that the run mode is "Run" not "Stop"
3. Verify sample rate matches your script settings

### High latency / dropped samples

1. Reduce the number of enabled channels
2. Increase buffer size in IntanRHXDevice
3. Check for other processes using CPU/network

## API Reference

### Basic Usage Pattern

```python
from intan.interface import IntanRHXDevice

# Use context manager for automatic cleanup
with IntanRHXDevice(sample_rate=20000, num_channels=128) as device:
    # Enable channels
    device.enable_wide_channel(range(32))  # Enable first 32 channels
    
    # Optional: Configure data blocks per write (affects latency/throughput)
    device.set_blocks_per_write(8)
    
    # Start streaming
    device.start_streaming()
    
    # Read latest data (non-blocking, returns None if no data available)
    data = device.get_latest_window(window_ms=200)  # Shape: (n_channels, n_samples)
    
    # Stop streaming when done
    device.stop_streaming()

# Device automatically closed when exiting context
```

### Key Methods

- **`enable_wide_channel(channel_list)`**: Enable wideband channels (0-30 kHz)
- **`enable_high_channel(channel_list)`**: Enable high-pass channels (250 Hz - 7.5 kHz)
- **`enable_low_channel(channel_list)`**: Enable LFP channels (0-1 kHz)
- **`set_blocks_per_write(n)`**: Configure data block buffering (higher = more latency, less CPU)
- **`start_streaming()` / `stop_streaming()`**: Control data acquisition
- **`get_latest_window(window_ms)`**: Non-blocking read of recent data
- **`record(duration_sec)`**: Blocking record for specified duration

### Data Format

All returned data follows the convention:
```python
data.shape  # (num_channels, num_samples)
data.dtype  # float64, units are microvolts (µV)
```
