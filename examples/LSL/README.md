# LSL Streaming Examples

This directory contains examples for real-time EMG visualization and processing using [Lab Streaming Layer (LSL)](https://labstreaminglayer.org/).

## Prerequisites

1. **Install pylsl**:
   ```bash
   pip install pylsl
   ```

2. **An active LSL stream**: These examples require an LSL source to be running. Options include:
   - Intan RHX with LSL output enabled
   - The `rhx_connect_to_device.py` script with `--use_lsl` flag
   - Any LSL-compatible acquisition software

## Examples

### lsl_stacked_plot.py

Real-time multichannel waveform viewer with stacked display.

```bash
# View channels 0-7 from an EMG stream
python lsl_stacked_plot.py --channels 0:8 --stream-type EMG

# View all channels from an EEG stream
python lsl_stacked_plot.py --channels all --stream-type EEG

# View specific channels
python lsl_stacked_plot.py --channels 0 5 10 15
```

**Features:**
- Stacked waveform display for multiple channels
- Automatic Y-axis scaling per channel
- Configurable time window
- Uses `LSLClient` for efficient multichannel buffering

### lsl_rms_barplot.py

Real-time RMS power visualization with adaptive channel quality control (QC).

```bash
# Basic usage with 128 channels
python lsl_rms_barplot.py --channels 128 --stream-type EMG

# Custom parameters
python lsl_rms_barplot.py --channels 64 --fs 2000 --window-ms 250 --ymax 1000
```

**Features:**
- RMS computed over sliding windows with adaptive quality checks
- Color-coded bars: blue (good), orange (watch), red (excluded)
- Automatic detection of bad channels (flat, noisy, power line interference)
- Keyboard shortcuts: `C` to calibrate baseline, `E` to export excluded channels
- Exports `excluded_channels.json` for downstream processing

### lsl_waveform_viewer.py

High-performance waveform viewer with pyqtgraph backend.

```bash
# View channels 0-3 with 2-second window
python lsl_waveform_viewer.py --channels 0-3 --win 2.0 --type EMG

# View with downsampling for rendering performance
python lsl_waveform_viewer.py --channels 0-7 --win 5.0 --downsample 2

# Specify stream by name or source_id
python lsl_waveform_viewer.py --name "IntanEMG" --channels 0,5,10
```

**Features:**
- Modern `LSLSubscriber` API with `LSLStreamSpec`
- Efficient ring buffer for smooth scrolling
- Downsample option for high-channel-count displays
- Keyboard shortcuts: `Q` or `Esc` to quit

### lsl_marker_sub.py

Subscribe to an LSL marker stream and print events to console.

```bash
# Subscribe to default "Markers" type stream
python lsl_marker_sub.py

# Subscribe to specific stream by name
python lsl_marker_sub.py --name "EMGGesture" --type Markers

# Increase timeout for slower networks
python lsl_marker_sub.py --timeout 10.0 --verbose
```

**Features:**
- Lightweight marker/event monitoring
- Background callback pattern for non-blocking operation
- Uses modern `LSLSubscriber` API

**Use cases:**
- Monitor gesture predictions from real-time classifier
- Track experiment events/triggers
- Debug LSL marker timing

## Creating an LSL Source

If you don't have an LSL source, you can create one from Intan hardware:

```bash
# From the interface examples directory
python rhx_connect_to_device.py --use_lsl --channels 128 --verbose
```

Or use a simulated source for testing:
```python
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time

info = StreamInfo('TestEMG', 'EMG', 8, 1000, 'float32', 'test123')
outlet = StreamOutlet(info)

while True:
    sample = (np.random.randn(8) * 50).tolist()
    outlet.push_sample(sample)
    time.sleep(0.001)
```

## Troubleshooting

### "No LSL stream found"

1. Check that your LSL source is running
2. Verify stream type matches (EMG, EEG, etc.)
3. List available streams:
   ```python
   from pylsl import resolve_streams
   streams = resolve_streams()
   for s in streams:
       print(f"{s.name()} - {s.type()} - {s.channel_count()} ch")
   ```

### High latency / choppy display

- Reduce the number of channels being plotted
- Increase the update interval
- Check CPU usage of the source application

### "Could not import pylsl"

Install with: `pip install pylsl`

On Linux, you may also need:
```bash
sudo apt-get install liblsl-dev
```

## Architecture

```
┌─────────────────┐      LSL       ┌─────────────────┐
│  Intan RHX      │  ──────────▶  │  LSLClient      │
│  (data source)  │    network    │  (subscriber)   │
└─────────────────┘               └────────┬────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │  Visualization  │
                                  │  (PyQt/pyqtgraph)│
                                  └─────────────────┘
```

LSL provides:
- Network transparency (stream from different machines)
- Time synchronization
- Automatic buffering
- Multiple subscriber support
