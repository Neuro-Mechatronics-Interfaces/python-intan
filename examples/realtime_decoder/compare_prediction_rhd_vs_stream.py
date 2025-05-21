import os
import numpy as np
import joblib
import tensorflow as tf
from intan.io import load_rhd_file, load_labeled_file
from intan.rhx_interface import IntanRHXDevice
from intan.processing import extract_features, notch_filter, bandpass_filter, lowpass_filter, rectify

def apply_filters(emg, fs):
    # These should match the training data
    emg = notch_filter(emg, fs, 60)
    emg = bandpass_filter(emg, 20, 450, fs)
    emg = rectify(emg)
    return lowpass_filter(emg, 5, fs)


def run_windowed_predictions(emg, fs, label_names, source):
    window_size = int(fs * (WINDOW_MS / 1000))
    step_size = int(fs * (STEP_MS / 1000))

    n_windows = (emg.shape[1] - window_size) // step_size + 1

    emg = apply_filters(emg, fs)
    correct = 0
    total = 0

    print(f"\n=== {source} ===")
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window = emg[:, start:end]

        feature_list =['mean_absolute_value', 'zero_crossings', 'slope_sign_changes', 'waveform_length', 'root_mean_square']
        feats = extract_features(window, feature_list)  # Default features
        feats = (feats - mean) / std
        feats = pca.transform(feats.reshape(1, -1))

        pred = model.predict(feats, verbose=0)
        gesture_idx = np.argmax(pred)
        gesture_name = label_names[gesture_idx]
        confidence = np.max(pred)

        # === Ground truth label from notes ===
        label = get_label_for_sample(start)
        if label and str(label).lower() != "nan":
            label = label.strip().lower()
            total += 1
            try:
                true_idx = next(i for i, name in enumerate(label_names) if label in name.lower())
            except StopIteration:
                true_idx = None

            if true_idx is not None and gesture_idx == true_idx:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            print(
                f"{status} [{start / fs:.2f}-{end / fs:.2f}s] → {gesture_name} | True: {label_names[true_idx] if true_idx is not None else label} | Conf: {confidence:.2f}")
        else:
            print(f"⚠️  [{start / fs:.2f}-{end / fs:.2f}s] → {gesture_name} | No ground truth | Conf: {confidence:.2f}")

    if total > 0:
        acc = correct / total
        print(f"\n🧪 Accuracy for {source}: {correct}/{total} ({acc:.2%})")


def get_label_for_sample(sample_idx):
    # Helper function for label lookup for any sample index
    for i in range(len(cue_df) - 1):
        start = cue_df.loc[i, "Sample"]
        end = cue_df.loc[i + 1, "Sample"]
        label = str(cue_df.loc[i, "Label"])
        if label.lower() != "none" and start <= sample_idx < end:
            return label
    return None


# === Config ===
DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan"
FILE_PATH = os.path.join(DATA_PATH, r"2024_10_22\raw\RingFlexion_241022_144153\RingFlexion_241022_144153.rhd")
MODEL_PATH = os.path.join(DATA_PATH, "model.keras")
PCA_PATH = os.path.join(DATA_PATH, "training_data_pca.pkl")
NORM_PATH = os.path.join(DATA_PATH, "training_data_norm.npz")
LABEL_PATH = os.path.join(DATA_PATH, "training_data.npz")
NOTES_PATH = os.path.join(os.path.dirname(FILE_PATH), "notes.txt")

WINDOW_MS = 250  # Make sure this matches what was used to create the training data
STEP_MS = 50  # Make sure this matches what was used to create the training data

TCP_DURATION_SEC = 10  # Configurable


# === Load model, PCA and normalization ===
model = tf.keras.models.load_model(MODEL_PATH)
pca = joblib.load(PCA_PATH)
norm = np.load(NORM_PATH)
label_names = np.load(LABEL_PATH, allow_pickle=True)["label_names"]
mean, std = norm["mean"], norm["std"]
std[std == 0] = 1  # Avoid division by zero

# === Load label timing file ===
cue_df = load_labeled_file(NOTES_PATH)

# 1) ========= Load and predict RHD file ============
#rhd = load_rhd_file(FILE_PATH)
#rhd_fs = rhd["frequency_parameters"]["amplifier_sample_rate"]
#rhd_emg = rhd["amplifier_data"][:, :int(rhd_fs * TCP_DURATION_SEC)]
#print(f"Shape of loaded data: {rhd_emg.shape}")
#run_windowed_predictions(rhd_emg, rhd_fs, label_names, source="RHD")

# 2) ======= Stream and predict from RHX device =======
device = IntanRHXDevice()
if not device.connected:
    raise RuntimeError("❌ Could not connect to RHX TCP server. Please start the Control GUI server.")

device.configure(channels=list(range(128)), blocks_per_write=1, enable_wide=True)
device.flush_commands()
tcp_fs = device.get_sample_rate()
ts, tcp_emg = device.stream(duration_sec=TCP_DURATION_SEC)  #n_frames=int(tcp_fs * TCP_DURATION_SEC))
print(f"Shape of streamed data: {tcp_emg.shape}")
device.close()

# Plot the 10 seconds of data from channel 5
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(ts, tcp_emg[5], label="TCP Stream", linewidth=1)
plt.title(f"Channel 5 — First {TCP_DURATION_SEC} Seconds")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

run_windowed_predictions(tcp_emg, tcp_fs, label_names, source="TCP")

# These 2 methods produce the exact same output, meaning the data is identical and the prediction performs as expected
