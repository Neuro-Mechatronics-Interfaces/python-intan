![Logo](https://raw.githubusercontent.com/neuro-mechatronics-interfaces/python-intan/main/docs/figs/logo.png)

# Python Intan

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://neuro-mechatronics-interfaces.github.io/python-intan/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/python-intan.svg)](https://badge.fury.io/py/python-intan)
[![Downloads](https://pepy.tech/badge/python-intan)](https://pepy.tech/project/python-intan)

**python-intan** is a comprehensive Python package for working with Intan Technologies RHX systems and electrophysiology data. From file loading to real-time streaming, signal processing to machine learning, hardware integration to GUI applications—everything you need for EMG/neural data analysis in one package.

---

## ✨ Key Features

- 📁 **File I/O**: Load `.rhd`, `.rhs`, `.dat`, `.csv`, and `.npz` files with ease
- 🔴 **Real-time Streaming**: TCP interface for live data acquisition from RHX devices
- 🎛️ **Signal Processing**: Filtering, normalization, RMS, feature extraction
- 🤖 **Machine Learning**: Complete gesture classification pipeline with TensorFlow
- 📊 **Visualization**: Waterfall plots, real-time plotting, GUI applications
- 🔌 **Hardware Integration**: LSL support, Raspberry Pi Pico, robotic control
- 🖥️ **GUI Applications**: EMG viewer, trial selector, gesture pipeline interface
- 🚀 **Performance**: GPU acceleration, optimized for real-time applications

---

## 📚 Quick Links

- [**Documentation**](https://neuro-mechatronics-interfaces.github.io/python-intan/) - Full guides and API reference
- [**Examples**](https://github.com/Neuro-Mechatronics-Interfaces/python-intan/tree/main/examples) - 60+ code examples
- [**FAQ**](https://neuro-mechatronics-interfaces.github.io/python-intan/info/faqs.html) - Frequently asked questions
- [**Contributing**](CONTRIBUTING.md) - How to contribute
- [**Changelog**](CHANGELOG.md) - Version history

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install python-intan
```

### From Source (Latest Features)

```bash
git clone https://github.com/Neuro-Mechatronics-Interfaces/python-intan.git
cd python-intan
pip install -e .
```

### With Virtual Environment

```bash
# Using conda
conda create -n intan python=3.10
conda activate intan
pip install python-intan

# Or using venv
python -m venv intan
source intan/bin/activate  # Windows: intan\Scripts\activate
pip install python-intan
```

### Optional Features

**GPU Support** - For faster machine learning training:
```bash
pip install tensorflow[and-cuda] nvidia-cudnn-cu12
```

**Video Processing** - For hand landmark tracking and finger kinematics:
```bash
pip install 'python-intan[video]'
```
Includes: opencv-python, mediapipe, and [handtrack](https://github.com/Jshulgach/Hand-Landmark-Tracker) package.

---

## 🚀 Getting Started

### Load and Visualize EMG Data

```python
import intan

# Load .rhd file (opens file picker)
result = intan.io.load_rhd_file()

# Or specify path directly
result = intan.io.load_rhd_file('path/to/file.rhd')

# Access data
emg_data = result['amplifier_data']  # Shape: (channels, samples)
fs = result['frequency_parameters']['amplifier_sample_rate']
t = result['t_amplifier']

# Quick filtering
emg_filtered = intan.processing.notch_filter(emg_data, fs, f0=60)
emg_filtered = intan.processing.filter_emg(emg_filtered, 'bandpass', fs,
                                            lowcut=10, highcut=500)

# Visualize
intan.plotting.waterfall(emg_filtered, range(64), t,
                         plot_title='Filtered EMG Data')
```

### Real-time Streaming from RHX Device

```python
from intan.interface import IntanRHXDevice

# Connect to device
device = IntanRHXDevice()
device.enable_wide_channel(range(64))
device.start_streaming()

# Stream data
timestamps, data = device.stream(duration_sec=1.0)
print(f"Acquired data shape: {data.shape}")

device.close()
```

### Train a Gesture Classifier

```python
from intan.ml import ModelManager
import numpy as np

# Load training data
data = np.load('training_data.npz')
X_train, y_train = data['features'], data['labels']

# Train model
manager = ModelManager()
model, pca, scaler = manager.train_model(X_train, y_train,
                                          model_type='CNN', epochs=50)

# Save for later use
manager.save_model('gesture_model.keras')
```

### Real-time Gesture Recognition

```python
from intan.interface import IntanRHXDevice
from intan.ml import EMGRealTimePredictor
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('gesture_model.keras')

# Initialize device
device = IntanRHXDevice(num_channels=128)
device.start_streaming()

# Create predictor
predictor = EMGRealTimePredictor(device, model, pca, mean, std, label_names)
predictor.run_prediction_loop()

# Get predictions
while True:
    prediction = predictor.get_prediction()
    if prediction:
        print(f"Gesture: {prediction['label']} ({prediction['confidence']:.1%})")
```

### Lab Streaming Layer (LSL) Integration

```python
from intan.interface import IntanRHXDevice, LSLPublisher

# Start device
device = IntanRHXDevice(num_channels=64)
device.start_streaming()

# Publish to LSL
publisher = LSLPublisher(name='IntanEMG', stream_type='EMG',
                         channel_count=64, sample_rate=4000)

while True:
    _, data = device.stream(n_frames=40)
    publisher.push_chunk(data.T)
```

---

## 🗂️ Package Structure

```text
intan/
├── io/                     # File loading (.rhd, .dat, .csv, .npz)
├── interface/              # RHX device, LSL, hardware interfaces
├── processing/             # Signal processing and filtering
├── ml/                     # Machine learning pipeline
├── plotting/               # Visualization utilities
├── applications/           # GUI applications
├── decomposition/          # PCA, ICA decomposition
└── samples/                # Sample data utilities

examples/
├── Read_Files/             # File loading examples
├── RHXDevice/              # Device streaming examples
├── LSL/                    # Lab Streaming Layer examples
├── gesture_classifier/     # ML training and prediction
├── applications/           # GUI application demos
├── 3D_printed_arm_control/ # Robotic control integration
└── interface/              # Hardware interfacing examples
```

---

## 🎯 Use Cases

### 📊 Data Analysis
- Load and analyze `.rhd` recordings
- Batch process multiple files
- Extract specific time segments
- Generate publication-quality figures

### 🔴 Real-time Applications
- Live EMG visualization
- Online gesture recognition
- Closed-loop control systems
- Synchronized multi-modal recording

### 🤖 Machine Learning
- Train gesture classifiers
- Real-time prediction
- Cross-session validation
- Transfer learning

### 🔬 Research
- Impedance testing
- Signal quality monitoring
- Protocol automation
- Custom experimental setups

---

## 📖 Documentation

Comprehensive documentation is available at [neuro-mechatronics-interfaces.github.io/python-intan](https://neuro-mechatronics-interfaces.github.io/python-intan/)

**Key Sections:**
- [Installation Guide](https://neuro-mechatronics-interfaces.github.io/python-intan/info/installation.html)
- [Loading Files](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/loading_data.html)
- [Real-time Streaming](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/live_plotting.html)
- [Signal Processing](https://neuro-mechatronics-interfaces.github.io/python-intan/info/signal_processing.html)
- [Gesture Classification](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/gesture_classification.html)
- [LSL Integration](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/lsl_streaming.html)
- [GUI Applications](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/gui_applications.html)
- [Hardware Control](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/hardware_control.html)
- [FAQ](https://neuro-mechatronics-interfaces.github.io/python-intan/info/faqs.html)
- [API Reference](https://neuro-mechatronics-interfaces.github.io/python-intan/intan_api/modules.html)

---

## 🎓 Examples

The `examples/` directory contains 60+ working examples organized by category:

```bash
# File loading
python examples/Read_Files/load_rhd_demo.py

# Real-time streaming
python examples/RHXDevice/scrolling_live.py

# Gesture classification pipeline
python examples/gesture_classifier/1a_build_training_dataset_rhd.py
python examples/gesture_classifier/2_train_model.py
python examples/gesture_classifier/3d_predict_from_device_realtime.py

# GUI applications
python examples/applications/run_emg_viewer.py
python examples/applications/gesture_pipeline_gui.py

# LSL streaming
python examples/LSL/lsl_waveform_viewer.py
python examples/LSL/lsl_rms_barplot.py
```

See the [Examples Documentation](https://neuro-mechatronics-interfaces.github.io/python-intan/examples/introduction.html) for complete guides.

---

## 🛠️ Supported Hardware

- **Intan RHX Controllers**: RHD USB Interface Board, RHD Recording Controller
- **Amplifiers**: RHD2000 series (RHD2132, RHD2164, RHD2216, etc.)
- **Stimulation**: RHS2000 series amplifiers
- **Peripherals**: Raspberry Pi Pico, servo controllers, IMU sensors
- **Integration**: Lab Streaming Layer (LSL) compatible devices

---

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 Bug reports
- ✨ Feature requests
- 📝 Documentation improvements
- 🧪 New examples
- 🔧 Code contributions

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Ways to contribute:**
- Report bugs or request features via [GitHub Issues](https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues)
- Submit pull requests with improvements
- Share your use cases and examples
- Help answer questions in discussions
- Improve documentation

---

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@software{Shulgach_Python_Intan_2025,
  author = {Shulgach, Jonathan and Murphy, Max and Foy, Adrian},
  title = {{Python Intan Package}},
  year = {2025},
  month = {01},
  version = {0.0.3},
  url = {https://github.com/Neuro-Mechatronics-Interfaces/python-intan},
  note = {Neuromechatronics Lab, Carnegie Mellon University}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License means:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

---

## 🙏 Acknowledgments

Developed by the [Neuromechatronics Lab](https://www.meche.engineering.cmu.edu/faculty/neuromechatronics-lab.html) at Carnegie Mellon University.

**Core Contributors:**
- Jonathan Shulgach
- Max Murphy
- Adrian Foy

**Special Thanks:**
- Intan Technologies for hardware and support
- The open-source neuroscience community

---

## 📧 Contact & Support

- **Documentation**: [neuro-mechatronics-interfaces.github.io/python-intan](https://neuro-mechatronics-interfaces.github.io/python-intan/)
- **Issues**: [GitHub Issues](https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues)
- **Email**: jshulgac@andrew.cmu.edu
- **Lab Website**: [Neuromechatronics Lab](https://www.meche.engineering.cmu.edu/faculty/neuromechatronics-lab.html)

---

## 🚦 Status & Roadmap

**Current Version: 0.0.3** (January 2025)

### Completed ✅
- [x] File loading (.rhd, .dat, .csv, .npz)
- [x] Real-time TCP streaming from RHX
- [x] Signal processing pipeline
- [x] Machine learning (CNN, LSTM, Dense models)
- [x] Real-time gesture recognition
- [x] Lab Streaming Layer integration
- [x] GUI applications (EMG viewer, trial selector, gesture pipeline)
- [x] Hardware integration (Pico, robotic arms, IMU)
- [x] Comprehensive documentation and examples

### In Progress 🚧
- [ ] Performance benchmarking suite
- [ ] Extended LSL marker synchronization
- [ ] Additional ML model architectures
- [ ] Mobile device integration

### Planned 📋
- [ ] Public training datasets
- [ ] Cloud integration for distributed processing
- [ ] Advanced impedance testing tools
- [ ] Multi-language support (MATLAB, Julia wrappers)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

<p align="center">
  <b>⭐ If you find this package useful, please consider giving it a star on GitHub! ⭐</b>
</p>

<p align="center">
  Made with ❤️ by the Neuromechatronics Lab
</p>
