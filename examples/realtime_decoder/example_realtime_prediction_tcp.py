import os
import time
import threading
import numpy as np
import joblib
import tensorflow as tf
from collections import deque
from intan.rhx_interface import IntanRHXDevice, config_options, DataStreamer
from intan.processing import extract_features, notch_filter, bandpass_filter, lowpass_filter, rectify


# === Config ===
WINDOW_MS = 250
N_CHANNELS = 128

# === Paths ===
DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan"
FILE_PATH = os.path.join(DATA_PATH, r"2024_10_22\raw\RingFlexion_241022_144153\RingFlexion_241022_144153.rhd")
MODEL_PATH = os.path.join(DATA_PATH, "model.keras")
PCA_PATH = os.path.join(DATA_PATH, "training_data_pca.pkl")
NORM_PATH = os.path.join(DATA_PATH, "training_data_norm.npz")
LABEL_PATH = os.path.join(DATA_PATH, "training_data.npz")
NOTES_PATH = os.path.join(os.path.dirname(FILE_PATH), "notes.txt")

# === Load model, PCA and normalization ===
model = tf.keras.models.load_model(MODEL_PATH)
pca = joblib.load(PCA_PATH)
norm = np.load(NORM_PATH)
label_names = np.load(LABEL_PATH, allow_pickle=True)["label_names"]
mean, std = norm["mean"], norm["std"]
std[std == 0] = 1

TRAINING_FEATURES = [
    'mean_absolute_value',
    'zero_crossings',
    'slope_sign_changes',
    'waveform_length',
    'root_mean_square'
]

# === Filters ===
def apply_filters(emg, fs):
    emg = notch_filter(emg, fs, 60)
    emg = bandpass_filter(emg, 20, 450, fs)
    emg = rectify(emg)
    return lowpass_filter(emg, 5, fs)

# === Connect to Intan device ===
device = IntanRHXDevice()
#config_options.set_blocks_per_write(device, 1)
config_options.set_channel(device, port='a', channel=0, enable_wide=True)
streamer = DataStreamer(device)
tcp_fs = config_options.get_sample_rate(device)
for ch in range(N_CHANNELS):
    config_options.set_channel(device, port='a', channel=ch, enable_wide=True)

# === Shared ring buffer for EMG data ===
SAMPLES_PER_WINDOW = int(WINDOW_MS / 1000 * tcp_fs)
BUFFER = deque(maxlen=SAMPLES_PER_WINDOW * 2)  # 2× window size
BUFFER_LOCK = threading.Lock()

# === Background thread for streaming ===
def stream_loop():
    print("Background streamer started...")
    while True:
        #try:
        _, emg = streamer.stream(n_frames=SAMPLES_PER_WINDOW)
        with BUFFER_LOCK:
            for i in range(emg.shape[1]):
                BUFFER.append(emg[:, i])
        #except Exception as e:
        #    print(f"[Streamer Error] {e}")
        #    break

# === Start background streaming thread ===
stream_thread = threading.Thread(target=stream_loop, daemon=True)
stream_thread.start()

# === Main prediction loop ===
print("🧠 Starting real-time inference on buffered EMG...\n")
try:
    while True:
        time.sleep(WINDOW_MS / 1000)

        with BUFFER_LOCK:
            if len(BUFFER) < SAMPLES_PER_WINDOW:
                continue
            emg_window = np.array(list(BUFFER)[-SAMPLES_PER_WINDOW:]).T  # [n_channels, window_len]

        filtered = apply_filters(emg_window, tcp_fs)
        feats = extract_features(filtered, feature_fns=TRAINING_FEATURES)
        feats = (feats - mean) / std
        reduced = pca.transform(feats.reshape(1, -1))
        pred = model.predict(reduced, verbose=0)
        idx = np.argmax(pred)
        gesture = label_names[idx]
        confidence = np.max(pred)

        print(f"[{time.strftime('%H:%M:%S')}] 🖐️ {gesture} | Conf: {confidence:.2f}")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    streamer.close()