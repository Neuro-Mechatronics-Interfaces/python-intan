# Changelog

All notable changes to the python-intan package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Performance benchmarking suite
- Extended LSL marker synchronization features
- Additional ML model architectures (transformers, attention mechanisms)
- Mobile device integration
- Public training datasets
- Cloud integration for distributed processing
- Advanced impedance testing tools

---

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
- `tensorflow==2.19.0`
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
