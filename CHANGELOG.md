# Changelog

All notable changes to the python-intan package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-08-14

### Added
- Console entry points for the EMG viewer and trial selector.
- Release-readiness regression tests and explicit test/build dependency group.
- Complete package metadata, classifiers, project URLs, and source manifest.

### Changed
- Split PyTorch, GUI, and video stacks into optional dependency groups.
- Standardized supported model training and inference on PyTorch; removed the legacy TensorFlow/Keras viewer path.
- Removed the cosmetic splash screen and its Pillow dependency.
- Reworked the README and maintained documentation around demonstrable workflows.
- Updated examples to run from the repository root and provide reliable `--help` output.
- Reduced wheel and source-archive size by removing unused and duplicated image assets.

### Fixed
- Repaired three syntax-broken examples and an unresolved merge marker.
- Restored lazy top-level subpackage access and optional plotting imports.
- Removed private-drive defaults, stale script references, and a cross-project package dependency.
- Corrected the gesture pipeline GUI's maintained script paths and command arguments.

---

## [0.1.0] - 2025-12-11

### Added
- **Configuration System**: New `intan.io._config_utils` module with GUI/terminal prompts
  - `load_simple_config()`, `save_simple_config()` for key=value config files
  - `prompt_directory()`, `prompt_file()`, `prompt_text()` with fallback to terminal
  - `get_or_prompt_value()` for CLI/config/prompt priority handling
- **Dataset Building**: New `intan.io._dataset_utils` module for ML workflows
  - `select_channels()` with channel mapping support
  - `process_recording()` for complete EMG→features→labels pipeline
  - `save_dataset()` with comprehensive metadata
- **HD-EMG Grid Support**: New `intan.io._grid_utils` module
  - `infer_grid_dimensions()` from channel names
  - `apply_grid_permutation()` for spatial transforms (rotation, flip, transpose)
  - `parse_orientation_from_filename()` for auto-detection
  - `remap_grid_channels()` for orientation correction
- **Device Channel Management**: New `intan.interface._device_utils` module
  - `normalize_channel_names()` for 0-based→1-based conversion
  - `parse_channel_spec()` for port-indexed dictionaries
  - `enable_channels_by_name()` for canonical naming
  - `build_active_channel_order()` for stream alignment
- **IMU Features**: New `intan.processing._imu_features` module
  - `aggregate_imu_features()` for multi-modal EMG+IMU datasets
  - `append_imu_features()` with zscore/robust normalization
  - Support for 'mean' and 'rich' (5-stat) feature modes
- **Prediction Pipelines**: New `intan.ml._prediction_modes` module
  - `predict_file()` for offline RHD file prediction
  - `predict_batch()` for multi-file aggregated metrics
  - `predict_from_device()` for fixed-duration recording
  - `predict_realtime_stream()` for continuous streaming
- **Prediction Utilities**: New `intan.ml._prediction_utils` module
  - `extract_features_from_emg()` with training-locked parameters
  - `compute_window_starts()` for event alignment
  - `predict_with_model()` with dimension validation
  - `predict_rhd_file()` for complete pipeline
- **Synchronization**: New `intan.processing._sync` module
  - `compute_landmark_movement_signal()` from MediaPipe hand tracking
  - `compute_emg_envelope_signal()` for alignment
  - `find_sync_offset()` via cross-correlation
  - `load_sync_offset()`, `save_sync_offset()` for persistence
- **Temporal Filters**: New `intan.processing._temporal_filters` module
  - `moving_average()`, `lowpass_filter()`, `savitzky_golay()`
  - `gaussian_smooth()`, `exponential_smooth()`, `adaptive_smooth()`
  - `smooth_predictions()` unified interface

- **Examples**: Finger Kinematics Pipeline Documentation
  - `CALIBRATION.md` - EMG normalization guide (session-to-session variability)
  - `SYNCHRONIZATION.md` - EMG-video alignment (cross-correlation based)
  - `WORKFLOW.md` - Complete 8-step pipeline reference
  - `README.md` - Streamlined quick start (reduced from 511→262 lines)

- **Examples**: Gesture Classifier Enhancements
  - `.gesture_config.example` - Shared config template for all scripts
  - `1_build_dataset.py` - Unified builder (RHD/NPZ/CSV, single/multi-file, 29+ params)
  - `3_predict.py` - Unified prediction CLI with 4 modes (file/batch/record/stream)
  - Interactive prompts with directory/file pickers
  - Poly5 file format support

- **Examples**: Interface Documentation
  - `examples/interface/README.md` - Hardware integration overview
  - LSL streaming examples
  - Raspberry Pi Pico firmware references

### Changed
- **Dataset Building**: Enhanced `1_build_dataset.py` with:
  - Smart event file discovery (checks `events/` subdirectory)
  - Visual separation of data (blue) and event (yellow) files in GUI
  - Improved error reporting for missing event files
  - Support for poly5 format alongside rhd/npz/csv
  - Channel mapping and orientation remapping
  - Multi-modal EMG+IMU features

- **Prediction**: Unified CLI in `3_predict.py`:
  - Single entry point for all prediction modes
  - Config file support (`.gesture_config`)
  - Interactive prompts when arguments missing
  - Shared parameter locking with training metadata

- **Documentation**: Finger kinematics workflow
  - Removed 6 redundant files (REFACTORING.md, QUICK_REFERENCE.md, demo scripts)
  - Consolidated step-by-step details into WORKFLOW.md
  - Added synchronization as critical first step (prevents 9s misalignment)

### Fixed
- **Event File Discovery**: Multi-file dataset building now correctly finds event files
  - Strips timestamps before matching (e.g., `file_251110_125854.rhd` → `file.event`)
  - Searches multiple patterns (`*_events.txt`, `*_event.txt`, `*.events`)
  - Prioritizes `events/` subdirectory structure

- **GUI**: Dataset builder file discovery improvements
  - Separate panels for data files (blue) vs event files (yellow)
  - Smart directory search (raw/ for data, events/ for labels)
  - Color-coded status (red=0 found, blue/brown=found)

- **Documentation**: Example organization
  - Removed test files and demos from finger_kinematics
  - Streamlined READMEs for clarity
  - Fixed hardcoded paths in gesture_pipeline_profile.json

### Removed
- Redundant documentation files from `examples/finger_kinematics/`:
  - `REFACTORING.md`, `REFACTORING_SUMMARY.md`, `QUICK_REFERENCE.md`
  - `demo_video_overlay.py`, `test_predict.py`
  - `finger_angle_display_demo.png`

## [0.0.3] - 2025-01-26

### Added
- **Documentation**: Comprehensive examples documentation covering 8 major categories
  - File loading and data access (`loading_data.rst`)
  - Gesture classification pipeline (`gesture_classification.rst`)
  - Lab Streaming Layer integration (`lsl_streaming.rst`)
  - GUI applications guide (`gui_applications.rst`)
  - Hardware control and integration (`hardware_control.rst`)
- **Documentation**: Expanded FAQ with 50+ questions across 11 categories
- **Documentation**: Created CONTRIBUTING.md with detailed contribution guidelines
- **Documentation**: Created CHANGELOG.md for version tracking
- **Examples**: Added CSV file loading support (`load_csv_demo.py`)
- **Examples**: LSL RMS bar plot visualization (`lsl_rms_barplot.py`)
- **Examples**: Cross-orientation gesture analysis script (`plot_cross_orientation.py`)
- **Examples**: Alternative model training pipeline (`2b_train_model.py`)
- **Processing**: Channel quality control utilities (`_channel_qc.py`)
- **IO**: CSV file utilities for loading preprocessed data (`_csv_utils.py`)

### Changed
- **Documentation**: Reorganized examples with categorized sections (Data Handling, Visualization, ML, Hardware)
- **Documentation**: Updated README with comprehensive feature list, use cases, and quick start examples
- **Documentation**: Improved API documentation structure with proper module organization
- **Module**: Renamed `intan.rhx_interface` → `intan.interface` for consistency
- **Examples**: Updated all import statements to use correct module names
- **Examples**: Fixed `config_options.set_channel()` → `device.enable_wide_channel()`
- **Version**: Updated version citations from 0.1.0 → 0.0.3 in README and CITATION.cff

### Fixed
- **Documentation**: Corrected autodoc module references in API docs
  - Fixed `intan.applications` submodule references
  - Fixed `intan.io` submodule names (added `_rhd_` prefix)
  - Fixed `intan.plotting` submodule references
- **Documentation**: Fixed code block syntax errors in `realtime_predict.rst`
- **Documentation**: Removed missing image references and replaced with documentation links
- **Build**: Removed incorrect dynamic version reference in `pyproject.toml`

### Documentation
- 4 new comprehensive example guides (60+ pages)
- 50+ FAQ entries with code examples
- Updated all cross-references to match new documentation structure
- Added table of contents to major documentation pages
- Improved code examples with proper error handling

---

## [0.0.2] - 2024-12-15

### Added
- **Lab Streaming Layer (LSL)**: Full integration support
  - LSL publisher for EMG data streams
  - LSL subscriber for marker streams
  - Synchronized recording capabilities
  - Real-time visualization from LSL streams
- **GUI Applications**: PyQt5-based applications
  - EMG Viewer with multiple display modes (waterfall, stacked, heatmap, RMS)
  - Trial Selector for data annotation and segmentation
  - Gesture Pipeline GUI for real-time gesture control
  - Application launcher for unified access
- **Machine Learning**: Real-time prediction
  - `EMGRealTimePredictor` class for online gesture recognition
  - Prediction confidence thresholds
  - Background prediction loops
  - Performance metrics (latency, accuracy)
- **Processing**: Additional signal processing functions
  - Channel quality assessment
  - Artifact detection
  - Advanced normalization methods
- **Interface**: Raspberry Pi Pico integration
  - Serial communication utilities
  - IMU data synchronization
  - Servo control for robotic applications

### Changed
- **Machine Learning**: Improved model training pipeline
  - Better PCA integration
  - Normalization parameter saving
  - Cross-validation support
- **Streaming**: Enhanced RHX device interface
  - Configurable buffer sizes
  - Improved error handling
  - Better connection diagnostics
- **Documentation**: Added real-time streaming guides
- **Examples**: Added 20+ new example scripts

### Fixed
- **Streaming**: Fixed buffer overflow in long-duration streaming
- **ML**: Corrected feature extraction for real-time windows
- **IO**: Improved .dat file parsing for edge cases

---

## [0.0.1] - 2024-05-02

### Added
- **Initial Release**: Core functionality for Intan data handling
- **File I/O**:
  - `.rhd` file loading and parsing
  - `.dat` file support
  - Header and metadata extraction
  - Channel information utilities
- **RHX Device Interface**:
  - TCP/IP streaming from RHX software
  - Command and data port communication
  - Channel configuration
  - Real-time data acquisition
- **Signal Processing**:
  - Bandpass, highpass, lowpass filtering
  - Notch filter (50/60 Hz powerline noise)
  - Rectification
  - RMS calculation
  - Downsampling
  - Normalization (z-score, min-max)
- **Machine Learning**:
  - PCA-based feature extraction
  - CNN model architecture for gesture classification
  - Training dataset builder from .rhd files
  - Model saving and loading utilities
- **Visualization**:
  - Waterfall plots for multi-channel EMG
  - Single-channel plotting
  - Real-time plotting utilities
  - Channel-by-name and channel-by-index plotting
- **Examples**:
  - File loading demos
  - RHX device streaming examples
  - Gesture classifier training pipeline
  - 3D-printed arm control integration
- **Documentation**:
  - Sphinx documentation setup
  - API reference
  - Basic installation guide
  - Getting started examples

### Dependencies
- `numpy>=1.26.0`
- `scipy==1.14.1`
- `matplotlib`
- `pandas`
- `PySerial`
- `scikit-learn`
- legacy neural-network runtime (removed from the supported stack in 0.2.1)
- `seaborn`
- `pyyaml`
- `PyQt5`
- `pyqtgraph`
- `pylsl`

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR version** (X.0.0): Incompatible API changes
- **MINOR version** (0.X.0): New features, backward compatible
- **PATCH version** (0.0.X): Bug fixes, backward compatible

**Current Status**: Pre-1.0 development phase
- API may change between minor versions
- Aiming for 1.0.0 release with stable API in 2025

---

## Release Process

1. Update CHANGELOG.md with new version and changes
2. Update version in:
   - `intan/__init__.py`
   - `pyproject.toml`
   - `CITATION.cff`
   - `README.md`
3. Build documentation: `cd docs && make html`
4. Run tests (if available): `pytest`
5. Build package: `python -m build`
6. Test on TestPyPI: `twine upload --repository testpypi dist/*`
7. Publish to PyPI: `twine upload dist/*`
8. Create GitHub release with tag `vX.Y.Z`
9. Update documentation website

---

## Links

- **Homepage**: https://github.com/Neuro-Mechatronics-Interfaces/python-intan
- **Documentation**: https://neuro-mechatronics-interfaces.github.io/python-intan/
- **PyPI**: https://pypi.org/project/python-intan/
- **Issues**: https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues

---

## Contributors

Thank you to all contributors who have helped improve python-intan!

**Core Team:**
- Jonathan Shulgach (@jshulgach)
- Max Murphy
- Adrian Foy

**Community Contributors:**
- [Your name here!](CONTRIBUTING.md)

---

*For older versions or detailed commit history, see [GitHub Releases](https://github.com/Neuro-Mechatronics-Interfaces/python-intan/releases).*
