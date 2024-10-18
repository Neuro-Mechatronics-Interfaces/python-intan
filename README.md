# Intan_EMG_Python
Tools and demos for working with EMG data from intan using python 

Also contains scripts for processing EMG data, extracting features, and training a machine learning classifier for gesture recognition based on sEMG signals.

Code was written and tested using Windows 11, Python 3.10.

## Repository Structure

* `1_preprocess_data.py` - Handles the creation of the data metrics file and trial event synchronization.
* `2_feature_extraction.py` - Extracts features from the EMG data. Currently supports dimensionality reduction using PCA.
* `3_train_model.py` - Trains an ML model for EMG classification, supporting Random Forest classifiers.
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
## Usage
1. Data Processing
    The first step is to preprocess the raw EMG data. We also need to create a metrics file that will store the information about trials, trial length, gestures, etc. Run the command:
    ```bash
    python 1_preprocess_data.py
    ```
2. Feature Extraction
    The next step is to extract features from the EMG data. This script currently supports PCA for dimensionality reduction. Run the command:
    ```bash
    python 2_feature_extraction.py
    ```
3. Train Model
    The final step is to train a machine learning model for gesture recognition. This script currently supports Random Forest classifiers. Run the command:
    ```bash
    python 3_train_model.py
    ```
   

## Future Improvements
 -  Add support for other classifiers
 -  Expand feature extraction to support CNN architectures.
 -  Add support for real-time classification using the trained models.
 -  Integrate with the Intan RHX system via TCP for real-time data streaming. - 
