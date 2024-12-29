# Real-time Decoder

This repository contains scripts for performing real-time classification of electromyography (EMG) data using a trained machine learning model. The scripts are designed to work with data collected using the Intan RHX system and the Intan Waveform Generator software. The real-time classification is performed using a trained model that was saved using the `gesture_classifier` scripts.

## Setup

1. If not done already, clone the repository and navigate to the `realtime_decoder` directory:
    ```bash
    git clone
    cd Intan_EMG_Python/realtime_decoder
    ```
2. Get the trained model file generated from the `gesture_classifier` directory (or your own) and update the `config.txt` file with the path to the model.
    - Note: be sure to pass the path of the config file as an argument during the training process. 
3. To perform real-time classification of EMG data, you need to have an Intan system connected to your computer with the sampling rate set to 4kHz. Enable the "Waveform Output" and "Commands" servers from `Remote TCP Control` tab with default settings:

![Remote_TCP.png](/../assets/Remote_TCP.png)
![Commands.png](/../assets/Commands.png)

## Real-time Classification
The real-time classification of gestures from EMG data is performed using the `realtime_decode.py` script. This script reads EMG data from the Intan system via TCP/IP and classifies the data using a trained model. The classification results are displayed in the terminal:
```bash
python realtime_decode.py
``` 
