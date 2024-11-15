# Intan_EMG_Python

This repository provides a set of tools and demonstrations for working with electromyography (EMG) data collected using Intan systems. It includes scripts for data preprocessing, feature extraction, machine learning (ML) model training, and real-time classification. These tools are designed to facilitate gesture recognition from sEMG signals, which can be applied in prosthetics, robotics, and neuromuscular research.

Code was written and tested using Windows 11, Python 3.10.
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Repository Structure

* `1_preprocess_data.py` - Handles the creation of the data metrics file and trial event synchronization.
* `2_feature_extraction.py` - Extracts features from the EMG data. Currently supports dimensionality reduction using PCA.
* `3_train_model.py` - Trains an ML model for EMG classification, supporting Random Forest classifiers.
* `4_realtime_decode.py` - Real-time classification of EMG data with connected INtan recorder using a trained model.
* `load_rhd_demo.py` - Opens a .rhd file and displays the EMG waveforms.

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
To open a .rhd file and display the EMG waveforms, run the following command:
```bash
python load_rhd_demo.py
```

## Classification Model Training
 The `config.txt` has a template of the contents that you should set up before running the scripts. The one that must be set up which defines where all other files are saved to is:
- `root_directory` - The path to the root directory containing the data.

There are optional naming parameters for file names which can be defined as well:
- `metrics_filename` - (default: "raw_data_metrics.csv") The name of the metrics file that will be created.
- `processed_data_filename` - (default: "processed_data.csv") The name of the file that will store the preprocessed data.
- `model_filename` - (default: "model.keras") The name of the file that will store the trained model.
- `gesture_label_filename` - (default: "gesture_labels.csv") The name of the file that will store the gesture labels.

1. The first step is to preprocess the raw EMG data. We also need to create a metrics file that will store the information about trials, trial length, gestures, etc. Set the path in the file and run in the terminal:
    ```bash
    python 1_preprocess_data.py --config_path=/mnt/c/Users/NML/path/to/config.txt
    ```
    - Note: If you keep your `config.txt` in the same directory as your scripts you can just run the script without having to define the path to the config file:
      ```bash
      python 1_preprocess_data.py
      ```
2. The next step is to extract features from the EMG data. This script currently supports PCA for dimensionality reduction. Set paths and run the command:
    ```bash
    python 2_feature_extraction.py
    ```
3. The final step is to train a machine learning model for gesture recognition. Set paths and run the command:
    ```bash
    python 3_train_model.py
    ```
    - Other arguments can be passed to the script to change the model type, number of components for PCA, etc. Use the `--help` flag to see all available options:
      ```bash
      python 3_train_model.py --help
      ```
   
## Real-time Classification
To perform real-time classification of EMG data, you need to have an Intan system connected to your computer with the sampling rate set to 4kHz. Enable the "Waveform Output" and "Commands" servers from `Remote TCP Control` tab with default settings and run:
```bash
python 4_realtime_decode.py
``` 

   

## Future Improvements
 - [x] Add support for other classifiers
 - [x] Expand feature extraction to support CNN architectures.
 - [x] Add support for real-time classification using the trained models.
 - [x] Integrate with the Intan RHX system via TCP for real-time data streaming.
 - [ ] Integrate support for sending serial commands to operate robot arm in realtime. 
 - [ ] Refine realtime classification to include a GUI for visualizing the data.
 - [ ] Allow downloading of dataset to perform actual training and testing.