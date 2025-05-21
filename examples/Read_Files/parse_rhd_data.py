from intan.io import load_rhd_file
import pandas as pd
import os
import numpy as np
from tqdm import tqdm



def load_gesture_timing(text_path):
    """
    Load gesture timing data from a text file.
    If the first row is a header (e.g., "Sample Index", "Timestamp", "Label"), it is skipped.
    """
    df_raw = pd.read_csv(text_path, header=None)

    # Check if the first row contains text headers instead of numbers
    if any(isinstance(cell, str) and "sample" in cell.lower() for cell in df_raw.iloc[0]):
        df_raw = df_raw[1:]  # Skip the first row if it's a header

    # Rename columns explicitly
    df_raw.columns = ["Sample", "Time", "Label"]

    # Remove rows with the word 'threshold' in the label
    df_raw = df_raw[~df_raw["Label"].astype(str).str.contains("threshold", case=False, na=False)]

    # Clean up and convert
    df_raw["Label"] = df_raw["Label"].astype(str).str.replace(" cue", "", case=False)
    df_raw["Sample"] = df_raw["Sample"].astype(int)

    # Sort by sample number
    df_sorted = df_raw.sort_values(by="Sample").reset_index(drop=True)

    return df_sorted


def extract_emg_segments_variable_length(emg_data, cue_df, include_rest=False):
    segments = []
    labels = []

    for i in tqdm(range(len(cue_df) - 1), desc="Extracting variable-length segments"):
        start = cue_df.iloc[i]["Sample"]
        end = cue_df.iloc[i + 1]["Sample"]
        label = cue_df.iloc[i]["Label"]

        # Skip NaN or "nan" labels
        if pd.isna(label) or str(label).strip().lower() == "nan":
            continue

        # Skip rest periods if requested
        if not include_rest and str(label).strip().lower() == "rest":
            continue

        if end > emg_data.shape[1]:
            continue

        segment = emg_data[:, start:end]
        segments.append(segment)
        labels.append(label)

    return segments, labels


def process_rhd_file(rhd_path, output_root):
    folder_path = os.path.dirname(rhd_path)
    folder_name = os.path.basename(folder_path)

    cue_file = os.path.join(folder_path, "notes.txt")
    if not os.path.exists(cue_file):
        print(f"⚠️ Skipping {folder_name}: notes.txt not found.")
        return

    result = load_rhd_file(rhd_path)
    print("got result data")
    cue_df = load_gesture_timing(cue_file)
    print(f"Found notes: {cue_df}")
    emg_data = result["amplifier_data"]

    segments, labels = extract_emg_segments_variable_length(emg_data, cue_df, include_rest=True)
    print(f"📈 Extracted {len(segments)} segments from {folder_name}")

    for i, (segment, label) in enumerate(zip(segments, labels)):
        file_safe_label = label.replace(" ", "_").lower()
        filename = f"{folder_name}_{file_safe_label}_{i}.npz"
        save_path = os.path.join(output_root, filename)
        np.savez_compressed(save_path, emg=segment, label=label)
        print(f"✅ Saved: {filename}")

def batch_process_rhd_dir(raw_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for folder_name in os.listdir(raw_root):
        folder_path = os.path.join(raw_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        rhd_file = os.path.join(folder_path, folder_name + ".rhd")
        if not os.path.exists(rhd_file):
            print(f"⚠️ No RHD file found in: {folder_path}")
            continue

        print(f"\n📂 Processing: {rhd_file}")
        process_rhd_file(rhd_file, output_root)

if __name__ == "__main__":
    RAW_DIR = r"path\to\data\folder\raw"
    OUT_DIR = os.path.join(os.path.dirname(RAW_DIR), "emg")
    batch_process_rhd_dir(RAW_DIR, OUT_DIR)
