
import os
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
import json

from intan.processing._filters import rectify, window_rms, z_score_norm, sliding_window

gesture_labels = [
    "handopen", "handclose", "fingersopen", "fingersclose",  # Can combine these
    "wristflexion", "wristextension",
    "indexflexion", "indexextension",
    "thumbflexion", "thumbextension",
    "middleflexion", "middleextension",
    "ringflexion", "ringextension",
    "pinkyflexion", "pinkyextension",
    "2fingerinch", "3fingerchuck",
    "rest"]
def extract_label(filename):
    base = os.path.basename(filename)
    parts = base.lower().split("_")
    for part in parts:
        if part in gesture_labels:
            # Renaming for consistency
            part = "handopen" if part == 'fingersopen' else part
            part = "handclose" if part == 'fingersclose' else part
            return part
    return "unknown"

def load_and_process_emg(npz_files):
    X_data = []
    y_labels = []

    for file in npz_files:
        data = np.load(file)
        emg = data["arr_0"]  # Expecting shape: (128, timepoints)

        if emg.shape[0] != 128:
            print(f"⚠️ Skipping {file}, expected 128 channels but got {emg.shape[0]}")
            continue

        label = extract_label(file)
        if label == "unknown":
            print(f"⚠️ Could not infer label for {file}")
            continue

        # === Jiang 2013 Preprocessing ===
        # 1. Rectify
        emg_rectified = rectify(emg)

        # 2. RMS envelope smoothing (200 ms window)
        emg_rms = window_rms(emg_rectified, window_size=200)

        # 3. Normalize per sample (z-score)
        emg_norm = z_score_norm(emg_rms)

        # 4. Segment into overlapping windows (200 ms, 50% overlap)
        windows = sliding_window(emg_norm, window_size=200, step_size=100)

        X_data.extend(windows)
        y_labels.extend([label] * len(windows))

    X = np.stack(X_data, axis=0)
    print(f"✅ Processed {X.shape[0]} windows | Shape: {X.shape}")
    return X, y_labels

def encode_labels(y_labels):
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    label_map = {label: int(le.transform([label])[0]) for label in le.classes_}
    return y, label_map

def save_dataset(X, y, label_map, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    with open(os.path.join(output_dir, "gesture_labels.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    print(f"✅ Saved to: {output_dir}")

if __name__ == '__main__':
    # Set the base directory for the subject
    subject_base = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan"
    session_folders = ["2024_10_22", "2024_11_11"]
    output_dir = os.path.join(subject_base, "processed_dataset")

    emg_dirs = [os.path.join(subject_base, session, "emg") for session in session_folders]
    all_npz_files = []

    # Show how many files are in each directory that need to be parsed
    print(f"🔍 Searching for .npz files in: {emg_dirs}")
    for d in emg_dirs:
        files = sorted(glob(os.path.join(d, "*.npz")))
        if not files:
            print(f"⚠️ No .npz files in {d}")
        else:
            print(f"📂 Found {len(files)} files in {d}")
            all_npz_files.extend(files)

    if not all_npz_files:
        print("🚫 No .npz files found. Exiting.")
        exit()



    X, y_labels = load_and_process_emg(all_npz_files)
    y, label_map = encode_labels(y_labels)
    save_dataset(X, y, label_map, output_dir)
