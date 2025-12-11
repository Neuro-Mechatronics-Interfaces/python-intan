# Python-Intan AI Coding Instructions

## Project Overview
**python-intan** is a comprehensive Python package for Intan Technologies RHX systems and electrophysiology data. It provides file I/O, real-time streaming, signal processing, machine learning for gesture classification, and hardware integration for EMG/neural data analysis.

## Architecture

### Module Structure
- **`intan.io`**: File loading (`.rhd`, `.rhs`, `.dat`, `.csv`, `.npz`) with canonicalized labels via `CANON` dict
- **`intan.interface`**: Real-time TCP streaming from RHX devices via `IntanRHXDevice` + LSL pub/sub
- **`intan.processing`**: Filtering, feature extraction, normalization, channel QC
- **`intan.ml`**: PyTorch models (`EMGClassifier`, `EMGRegressor`), `ModelManager` for training/loading
- **`intan.plotting`**: Waterfall plots, real-time visualization, GUI applications
- **`intan.applications`**: Full GUI apps (EMG viewer, trial selector, gesture pipeline)

### Key Design Patterns

#### 1. Context Manager for Hardware
Always use `IntanRHXDevice` with context managers to ensure proper cleanup:
```python
with IntanRHXDevice(sample_rate=4000, num_channels=128) as device:
    device.enable_wide_channel(range(128))
    emg = device.record(duration_sec=10)
```

#### 2. Configuration Hierarchy
`IntanRHXDevice` inherits from `RHXConfig` which wraps TCP commands. Configure via:
- `enable_wide_channel()` / `enable_high_channel()` / `enable_low_channel()`
- `set_run_mode()`, `set_sample_rate()`, `set_blocks_per_write()`
- TCP commands sent with `set_parameter()` / `get_parameter()`

#### 3. Feature Registry Pattern
Features use a **registry-based system** for extensibility (`intan.processing._features.py`):
```python
FEATURE_REGISTRY = {
    'rms': compute_rms,
    'mav': mean_absolute_value,
    'var': variance,
    # ... more features
}
```
Extract features via `extract_features(data, features=['rms', 'mav'])` or `extract_features_sliding_window()` for time-series.

#### 4. Label Canonicalization
All channel/label names go through `canonical_label()` using `CANON` dict in `intan.io._canonicalizer`. This ensures consistent naming across file formats (e.g., `'A-000'` → `'amplifier'`).

#### 5. Data Loading Patterns
- **Interactive**: `load_rhd_file()` opens file picker if no path provided
- **Batch**: `load_rhd_file(filepath, merge_files=True)` auto-concatenates multi-part recordings
- **Config-driven**: Use `load_config_file()` for YAML/JSON/TXT configs (see `examples/gesture_classifier/`)

## Development Workflows

### Running Examples
Examples are organized by category in `examples/`:
- **`RHXDevice/`**: Basic streaming/recording demos (start here)
- **`gesture_classifier/`**: Full ML pipeline (dataset → training → prediction)
- **`LSL/`**: Lab Streaming Layer integration
- **`interface/`**: Hardware integration (Raspberry Pi Pico, IMU)

Run examples from repo root:
```bash
python examples/RHXDevice/record_demo.py
python examples/gesture_classifier/2_train_model.py --config_path=path/to/config.txt
```

### Testing GPU Support
TensorFlow ML models support GPU acceleration:
```bash
pip install tensorflow[and-cuda] nvidia-cudnn-cu12
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Installation Modes
- **User**: `pip install python-intan`
- **Dev**: `pip install -e .` (editable mode, required for example scripts)
- **Docs**: `pip install -e .[docs]` (includes Sphinx dependencies)

## Project-Specific Conventions

### 1. Data Shape Convention
**EMG data is `(channels, samples)`** throughout the codebase:
```python
emg_data.shape  # (num_channels, num_samples)
result = processing.filter_emg(emg_data, ...)  # preserves shape
```

### 2. Sampling Rate Propagation
Always pass `fs` (sampling rate) explicitly to processing functions:
```python
fs = result['frequency_parameters']['amplifier_sample_rate']
filtered = processing.notch_filter(emg_data, fs, f0=60)
filtered = processing.bandpass_filter(filtered, fs, lowcut=10, highcut=500)
```

### 3. Channel Selection Patterns
Use integer lists or ranges for channel selection:
```python
device.enable_wide_channel(range(128))  # Not string names
device.enable_wide_channel([0, 5, 10])  # Specific channels
```

### 4. ML Dataset Structure
Training datasets follow a standard `.npz` format (see `examples/gesture_classifier/1a_build_training_dataset_rhd.py`):
```python
np.savez(output_path,
    X=features,                    # (n_samples, n_features)
    y=labels,                      # (n_samples,) strings
    y_id=label_ids,                # (n_samples,) integers
    class_names=class_names,
    label_to_id_json=json.dumps(label_to_id),
    feature_spec=json.dumps(feature_spec),
    emg_fs=fs,
    window_ms=window_ms,
    step_ms=step_ms
)
```

### 5. TCP Streaming Protocol
`IntanRHXDevice` uses two TCP connections:
- **Command socket (5000)**: Configuration via `RHXConfig` methods
- **Data socket (5001)**: Binary stream parsing with `parse_emg_stream_fast()`

Data arrives in blocks of `FRAMES_PER_BLOCK = 128` frames. Set `set_blocks_per_write(8)` for typical performance.

## Integration Points

### Lab Streaming Layer (LSL)
Use `LSLPublisher` / `LSLSubscriber` for multi-system data sync:
```python
from intan.interface import LSLNumericPublisher, LSLMarkerPublisher
emg_pub = LSLNumericPublisher("EMG", num_channels=128, fs=4000)
marker_pub = LSLMarkerPublisher("Events")
```

### Real-time Prediction
Use `EMGRealTimePredictor` for live gesture classification:
```python
from intan.ml import EMGRealTimePredictor
predictor = EMGRealTimePredictor(model_path="model.keras", config_path="config.json")
prediction = predictor.predict(emg_window)
```

### External Hardware
- **Raspberry Pi Pico**: See `examples/interface/pico/` for CircuitPython → TCP bridge
- **IMU Integration**: `examples/interface/rhx_emg_and_imu.py` for synchronized streams

## Common Pitfalls

1. **Channel Naming**: Use canonical labels (`'amplifier'`, not `'A-000'`). Check `CANON` dict if unsure.
2. **Data Shape**: Always verify `(channels, samples)` not `(samples, channels)` before processing.
3. **TCP Connection**: RHX software must be running with TCP servers enabled before connecting.
4. **File Paths**: Use `adjust_path()` from `intan.io` for cross-platform path handling.
5. **Memory**: Large `.rhd` files (>2GB) may require `merge_files=False` + manual concatenation.

## Key Files for Reference
- **Architecture**: `intan/__init__.py` (module exports), `intan/io/__init__.py` (file I/O API)
- **Device Control**: `intan/interface/_rhx_device.py`, `intan/interface/_rhx_config.py`
- **Processing**: `intan/processing/_features.py` (FEATURE_REGISTRY), `intan/processing/_filters.py`
- **ML Pipeline**: `examples/gesture_classifier/README.md`, `intan/ml/_model_manager.py`
- **Contributing**: `CONTRIBUTING.md` (dev setup, style guide, bug reporting)
