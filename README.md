# Python Intan

This repository provides a set of tools and demonstrations for working with electromyography (EMG) data collected using Intan systems. It includes scripts for data preprocessing, feature extraction, machine learning (ML) model training, and real-time classification. These tools are designed to facilitate gesture recognition from sEMG signals, which can be applied in prosthetics, robotics, and neuromuscular research.

![intan_logo.png](/assets/intan_logo.png)


Code was written and tested using Windows 11, Python 3.10.
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Repository Structure

* `3D_printed_arm_control` - Hardware and software resources for robot arm control using microcontroller supporting CircuitPython.
* `realtime_decoder` - Perform inference on an EMG signal in real-time using a trained model.
* `gesture_classifier` - Scripts for training and testing machine learning models for gesture classification.
* `utilities` - Helper functions for data preprocessing, feature extraction, and model evaluation.

## Installation
1. It is recommended to use a virtual environment to manage dependencies. To create a new virtual environment with [anaconda](https://www.anaconda.com/products/individual), use the following command:

   ```bash
   conda create -n intan python=3.10
   conda activate intan
   ```
2. Download the repository using git:
   ```bash
   git clone https://github.com/Neuro-Mechatronics-Interfaces/Intan_EMG_Python.git
   cd Intan_EMG_Python
   ```
3. To install dependencies, use the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```
   
## Demo
A demo script in the main directory shows a quick example of opening and plotting EMG waveforms from a .rhd file. Run the following command:
```bash
python load_rhd_demo.py
```


   

## Future Improvements
 - [x] Add support for other classifiers
 - [x] Expand feature extraction to support CNN architectures.
 - [x] Add support for real-time classification using the trained models.
 - [x] Integrate with the Intan RHX system via TCP for real-time data streaming.
 - [x] Integrate support for sending serial commands to operate robot arm in realtime. 
 - [ ] Refine realtime classification to include a GUI for visualizing the data.
 - [ ] Allow downloading of dataset to perform actual training and testing.
 - [ ] Allow data analysis methods to be used on the dataset.
