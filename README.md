# Python Intan

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Neuro-Mechatronics-Interfaces/python-intan/test.yml?branch=main)](https://github.com/Neuro-Mechatronics-Interfaces/python-intan/actions)
[![PyPI version](https://badge.fury.io/py/python-intan.svg)](https://badge.fury.io/py/python-intan)

<p align="center">
  <img src="./docs/figs/logo.png">
</p>


## Overview

**`python-intan`** is a Python package for working with EMG data acquired from Intan Technologies systems. It provides utilities for:

- Reading `.rhd` and `.dat` data files
- Preprocessing and feature extraction
- Integration with external hardware for closed-loop applications

Whether you're analyzing offline data or streaming real-time EMG from the RHX system, `python-intan` aims to make the workflow seamless and modular.

---

## Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Installation

Create a virtual environment using conda or venv:

```bash
conda create -n intan python=3.10
conda activate intan
```
or

```bash
python -m venv intan
source intan/bin/activate  # On Windows use: intan\Scripts\activate
```
(Optional) Enable CUDA support for GPU acceleration (if available):

```bash
conda install cudatoolkit cudnn=11
```
or 
```bash
sudo apt-get install cuda
pip install tensorflow[and-cuda] nvidia-cudnn-cu12 # CUDA 12.5 latest pip option as of 5/13/25
```

This repo can be installed from PyPi using `pip`:

```bash
pip install python-intan
```

or clone & install the repository from GitHub for the latest features:

```bash
git clone https://github.com/Neuro-Mechatronics-Interfaces/python-intan.git
cd python-intan
pip install -e .
```

---

## Getting Started

We can quickly load EMG signals from an Intan `.rhd` file. 

```python
import intan

result = intan.io.load_rhd_file() # Opens Popup window to select file, or pass filepath
```
`.dat` files are also supported
```python
dat_folderpath = "path/to/your/dat/files"
result = intan.io.load_dat_file(dat_folderpath)
```
The `result` dictionary contains:
- `amplifier_data`: Raw EMG data (shape: channels x samples)
- `amplifier_channels`: Channel metadata (names, gain, etc.)
- `frequency_parameters`: Sampling rate information for the amplifier, aux, and digital inputs
- `t_amplifier`: Time vector for the amplifier data
- `board_adc_data`: ADC data from the board

Loading multiple `.rhd` files with the option to concatenate can be done with:
```python
result_list = intan.io.load_rhd_files(concatenate=True)
```
Some quick filtering to remove DC offset and noise:
```python
fs = result['frequency_parameters']['amplifier_sample_rate']
emg = result['amplifier_data']  # Shape: (channels, samples)

notch_emg = intan.processing.notch_filter(emg, fs, f0=60) # 60Hz notch filter
filt_emg = intan.processing.filter_emg(notch_emg, 'bandpass', fs, lowcut=10, highcut=500) # 10-500Hz bandpass filter
```


We can visualize the raw EMG data using the `plotting` module. Let's use a waterfall plot to look at the raw activity across a subset of channels:

```python
t_s = result['t_amplifier']
chans = range(0, 64)  # Selecting channels to plot
intan.plotting.waterfall(filt_emg, chans, t_s, plot_title='Intan EMG data')
```

If there is a single channel of interest, we can plot it directly by name or index:
```python
intan.plotting.plot_channel_by_name('A-005', result)  # By name (Be sure the channel name exists)
intan.plotting.plot_channel_by_index(8, result)  # By index
```

Some other processing features:
```python
# Normalize the EMG data
norm_emg = intan.processing.normalize(filt_emg, method='zscore')  # Z-score normalization

# Downsample the EMG data
downsampled_emg = intan.processing.downsample(filt_emg, factor=2)  # Downsample by a factor of 2

# Rectify
rectified_emg = intan.processing.rectify(filt_emg)  # Rectify the EMG data

# Rolling RMS window
samples = int(0.1 * fs)  # 100ms window
rms_emg = intan.processing.window_rms(filt_emg, window_size=samples)  # 100ms RMS window
```


---

## Project Structure

```text
intan/
├── io/                 # Load and parse Intan .rhd or .rhs files
├── processing/         # Filtering, normalization, and segmentation
├── control/            # Real-time control scripts for robot arm or other peripherals
├── plotting/      # Plotting and GUI scripts

load_rhd_demo.py        # Quick-start script for plotting waveform data
requirements.txt
README.md
```

---

## Future Improvements

- [x] Real-time gesture decoding via TCP from RHX
- [x] Serial control for wearable robotics (e.g. grippers, arms)
- [x] CNN feature pipeline support
- [ ] GUI for real-time signal monitoring
- [ ] Public datasets with training/testing support
- [ ] Cross-platform support for macOS/Linux (currently optimized for Windows)

---

## Citation

This package can be cited as follows:

```bibtex
@software{Shulgach_Python_Intan_2025,
  author = {Shulgach, Jonathan, Murphy, Max and Foy, Adrian},
  title = {{Python Intan Package}},
  year = {2025},
  month = {05},
  version = {0.1.0},
  url = {https://github.com/Neuro-Mechatronics-Interfaces/python-intan},
  note = "{\tt jshulgac@andrew.cmu.edu}"
}
```


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
