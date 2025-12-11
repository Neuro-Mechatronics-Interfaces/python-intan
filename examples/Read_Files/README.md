# File I/O Examples

Examples for loading and parsing Intan data files.

## Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| RHD | `.rhd` | Intan traditional file format (header + data in one file) |
| RHS | `.rhs` | Intan stimulation file format |
| DAT | `.dat` + `.rhd` | One-file-per-signal-type format |
| NPZ | `.npz` | NumPy compressed archive (for preprocessed data) |
| CSV | `.csv` | Comma-separated values (third-party exports) |

## Examples

### load_rhd_demo.py

Load and visualize an Intan RHD file.

```bash
# Interactive file dialog
python load_rhd_demo.py

# Or specify path directly (edit the script)
python load_rhd_demo.py --save_npz  # Save as NPZ for faster future loads
```

**What you get:**
```python
result = load_rhd_file("/path/to/recording.rhd")

result['amplifier_data']     # EMG data: (n_channels, n_samples)
result['t_amplifier']        # Time vector: (n_samples,)
result['channel_names']      # ['A-000', 'A-001', ...]
result['frequency_parameters']['amplifier_sample_rate']  # e.g., 4000.0
result['board_adc_data']     # Analog inputs (if present)
result['board_dig_in_data']  # Digital inputs (if present)
```

### load_dat_demo.py

Load data from one-file-per-signal-type format.

```bash
python load_dat_demo.py --directory /path/to/recording_folder
```

**Expected folder structure:**
```
recording_folder/
├── info.rhd           # Header file
├── time.dat           # Timestamps
├── amplifier.dat      # EMG channels
├── auxiliary.dat      # Aux inputs (optional)
└── analogin.dat       # ADC inputs (optional)
```

### load_csv_demo.py

Load EMG data from CSV files.

```bash
python load_csv_demo.py --file /path/to/data.csv --fs 2000
```

### parse_rhd_data.py

Low-level RHD parsing example showing the file structure.

```bash
python parse_rhd_data.py --file /path/to/recording.rhd --verbose
```

### segment_emg_from_events.py

Extract labeled segments from a recording using an event file.

```bash
python segment_emg_from_events.py \
    --rhd_file /path/to/recording.rhd \
    --event_file /path/to/labels.event \
    --output_dir /path/to/segments
```

**Output:**
```
segments/
├── WristFlexion_001.npz
├── WristFlexion_002.npz
├── WristExtension_001.npz
└── ...
```

## File Format Details

### RHD Header Structure

The RHD file starts with a header containing:
- Magic number and version
- Sample rate settings
- Frequency band parameters
- Channel information (names, impedances)
- Notes and board configuration

```python
from intan.io import read_header

with open("recording.rhd", "rb") as f:
    header = read_header(f)
    
print(f"Sample rate: {header['sample_rate']} Hz")
print(f"Channels: {header['num_amplifier_channels']}")
```

### Data Blocks

After the header, data is stored in blocks:
- Block size: 128 samples per channel (RHD2000) or 128 frames
- Order: timestamps, then amplifier, then aux, then ADC, then digital

### NPZ Format

Our NPZ files store:
```python
np.savez(
    "recording.npz",
    amplifier_data=emg,           # (channels, samples)
    t_amplifier=t,                # (samples,)
    sample_rate=fs,               # scalar
    channel_names=names,          # list of strings
    # Optional metadata...
)
```

## Performance Tips

### Large Files

For files > 1 GB:
```python
# Load only specific channels
result = load_rhd_file("large.rhd", channels=[0, 1, 2, 3])

# Or load in chunks
from intan.io import read_one_data_block
# ... process block by block
```

### Merging Multiple Files

Intan saves files in segments (e.g., 60-second chunks):
```python
result = load_rhd_file("/path/to/folder", merge_files=True)
```

### NPZ for Speed

After first load, save as NPZ for 10x faster subsequent loads:
```python
from intan.io import load_rhd_file, save_as_npz

result = load_rhd_file("recording.rhd")
save_as_npz(result, "recording.npz")

# Future loads:
data = np.load("recording.npz", allow_pickle=True)
```

## Troubleshooting

### "Unrecognized file format"

- Verify the file is a valid Intan RHD/RHS file
- Check file isn't corrupted (incomplete recording)
- Ensure you're using `load_rhd_file` for RHD, `load_dat_file` for DAT

### "File size mismatch"

The file may be incomplete (recording was interrupted). Use:
```python
result = load_rhd_file("incomplete.rhd", repair=True)
```

### Memory errors with large files

Load subset of channels or use memory-mapped arrays:
```python
# Coming soon: memory-mapped loading
result = load_rhd_file("huge.rhd", mmap=True)
```
