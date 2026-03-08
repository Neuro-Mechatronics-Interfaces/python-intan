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
# Python-Intan: Copilot / AI Agent Instructions

Purpose: get an AI agent productive quickly — architecture, dev workflows,
key patterns, and concrete file examples.

- Quick start (dev): activate virtualenv then install editable package

    - PowerShell (Windows): `& .venv\Scripts\Activate.ps1`
    - Install: `pip install -e .`
    - Run examples: `python examples/RHXDevice/record_demo.py`

- Where to look first
    - Core API surface: [intan/__init__.py](intan/__init__.py)
    - File I/O: [intan/io](intan/io/__init__.py)
    - Device & TCP: [intan/interface/_rhx_device.py](intan/interface/_rhx_device.py), [intan/interface/_rhx_config.py](intan/interface/_rhx_config.py)
    - Processing features: [intan/processing/_features.py](intan/processing/_features.py)
    - Examples: `examples/` (gesture_classifier, RHXDevice, interface/pico)

- Big picture
    - Purpose: read Intan `.rhd`/.`rhs` files, stream from RHX devices, process EMG,
        and run ML pipelines for gesture classification and real-time prediction.
    - Data flow: device or file -> `intan.io` loader or `IntanRHXDevice` ->
        `intan.processing` (filters/features) -> `intan.ml` models / `intan.plotting`.

- Important, repo-specific conventions (must-follow)
    - EMG array shape: always (channels, samples). Many functions expect this.
    - Sampling-rate param: pass `fs` explicitly to filters/features.
    - Channel selection: use integer indices or ranges (e.g. `range(128)`), not names.
    - Feature registry: add new features to [intan/processing/_features.py](intan/processing/_features.py) via `FEATURE_REGISTRY`.
    - Label canonicalization: use the `CANON` mapping in `intan/io/_canonicalizer` for consistent channel names.

- Device integration notes
    - `IntanRHXDevice` is a context manager — always use `with` to ensure cleanup.
    - TCP protocol: command socket (5000) and data socket (5001). Data parsed in
        blocks (`FRAMES_PER_BLOCK = 128`) — see `_rhx_device` and `parse_emg_stream_fast()`.

- Development & testing tips
    - Run specific example scripts from project root; many expect editable install.
    - Docs build: `docs\\make.bat html` on Windows or `make -C docs html` on *nix.
    - Tests: `pytest -q` (use venv with dev deps installed).

- When editing code (style & safety)
    - Keep public APIs unchanged unless the change is necessary.
    - Preserve `(channels, samples)` shape through transforms — tests depend on it.
    - Prefer minimal, focused changes; update the feature registry and examples when adding features.

- Useful concrete examples to reference while coding
    - Stream + record demo: [examples/RHXDevice/record_demo.py](examples/RHXDevice/record_demo.py)
    - Build dataset example: [examples/gesture_classifier/1a_build_training_dataset_rhd.py](examples/gesture_classifier/1a_build_training_dataset_rhd.py)
    - Real-time EMG+IMU integration: [examples/interface/rhx_emg_and_imu.py](examples/interface/rhx_emg_and_imu.py)

If any section should be expanded (CLI, CI, or specific files to reference), tell me which area and I will iterate.
    label_to_id_json=json.dumps(label_to_id),
