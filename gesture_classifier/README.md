# Gesture Classifier

This repository contains scripts for training a machine learning model to classify gestures from EMG data. The scripts are written in Python and use the `numpy`, `pandas`, `scikit-learn`, and `keras` libraries.


## GPU Setup

The model training can work without the GPU but the training time increases exponentially. It is recommended to have a CUDA compatible GPU to train the model.

### Setup System

This readme and code were written and tested using WSL2 on Windows11, Python 3.10.
1. Install WSL2 on Windows 11
    - Follow the instructions on the [Microsoft website](https://learn.microsoft.com/en-us/windows/wsl/install). Skip if already installed.
1. Download NVIDIA drivers
    Follow the instructions on the [NVIDIA website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) to install CUDA on WSL2. Remember to select the `WSL-Ubuntu`, version 2.0.

Once the drivers are installed, you can check to see if the drivers were successfully installed using the command `nvidia-smi` in the terminal. 
```
Sat Dec 28 22:52:39 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.72                 Driver Version: 566.14         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 Ti     On  |   00000000:09:00.0  On |                  N/A |
|  0%   40C    P8             20W /  310W |     906MiB /   8192MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A       347      G   /Xwayland                                   N/A      |
+-----------------------------------------------------------------------------------------+
```
You can also verify that TensorFlow will work with the GPU by running the following code in a Python terminal:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))"
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
    python preprocess_data.py --config_path=/mnt/c/Users/NML/path/to/config.txt
    ```
    - Note: If you keep your `config.txt` in the same directory as your scripts you can just run the script without having to define the path to the config file:
      ```bash
      python preprocess_data.py
      ```
2. The next step is to extract features from the EMG data. This script currently supports PCA for dimensionality reduction. Set paths and run the command:
    ```bash
    python feature_extraction.py
    ```
3. The final step is to train a machine learning model for gesture recognition. Set paths and run the command:
    ```bash
    python train_model.py
    ```
    - Other arguments can be passed to the script to change the model type, number of components for PCA, etc. Use the `--help` flag to see all available options:
      ```bash
      python train_model.py --help
      ```
      