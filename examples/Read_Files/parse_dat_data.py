from intan.io import load_dat_file
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def load_gesture_timing(text_path):
    """Load gesture cues from a text file."""
    df = pd.read_csv(text_path, header=None, names=["Sample", "Time", "Label"])
    df["Label"] = df["Label"].str.replace(" cue", "", case=False)
    # Remove rows with the word threshold
    df = df[~df["Label"].str.contains("threshold", case=False)]
    df["Sample"] = df["Sample"].astype(int)
    return df

def extract_emg_segments(emg_data, cue_df, window_size=512):
    """
    Extract EMG windows starting from each gesture cue sample.
    - emg_data: np.array, shape (channels, samples)
    - cue_df: DataFrame with columns ['Sample', 'Time', 'Label']
    - window_size: number of samples per segment
    """
    segments = []
    labels = []

    for _, row in tqdm(cue_df.iterrows(), total=len(cue_df)):
        start = row["Sample"]
        end = start + window_size
        if end > emg_data.shape[1]:
            continue  # Skip if beyond available data
        segment = emg_data[:, start:end]
        segments.append(segment)
        labels.append(row["Label"])

    return np.stack(segments), labels


def extract_emg_segments_variable_length(emg_data, cue_df, include_rest=False):
    """
    Extract EMG segments between cue changes.

    Args:
        emg_data (np.ndarray): Shape (channels, samples)
        cue_df (pd.DataFrame): Must contain ['Sample', 'Label']
        include_rest (bool): Whether to include rest periods

    Returns:
        segments (List[np.ndarray]): List of EMG arrays [channels, duration]
        labels (List[str]): Corresponding gesture labels
    """
    segments = []
    labels = []

    for i in tqdm(range(len(cue_df) - 1), desc="Extracting variable-length segments"):
        start = cue_df.iloc[i]["Sample"]
        end = cue_df.iloc[i + 1]["Sample"]
        label = cue_df.iloc[i]["Label"]

        if not include_rest and label.lower() == "rest":
            continue
        if end > emg_data.shape[1]:
            continue

        segment = emg_data[:, start:end]
        segments.append(segment)
        labels.append(label)

    return segments, labels


def batch_process_emg_sessions(raw_root, output_root, window_size=512):
    """Process all gesture folders in a raw directory and save EMG slices per gesture."""
    os.makedirs(output_root, exist_ok=True)

    for folder_name in os.listdir(raw_root):
        folder_path = os.path.join(raw_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        #if folder_name != "MVCRest_250321_164238":
        #    continue

        print(f"\n📂 Processing folder: {folder_name}")
        try:
            result = load_dat_file(folder_path)
            cue_file = os.path.join(folder_path, "notes.txt")
            if not os.path.exists(cue_file):
                print(f"⚠️ Skipping: notes.txt not found in {folder_name}")
                continue

            cue_df = load_gesture_timing(cue_file)
            emg_data = result["amplifier_data"]
            segments, labels = extract_emg_segments_variable_length(emg_data, cue_df, include_rest=True)
            print(f"Extracted {len(segments)} segments.")
            #print(f"Segment shape: {segments.shape}")
            print(f"Labels: {labels}")

            for i, (segment, label) in enumerate(zip(segments, labels)):
                file_safe_label = label.replace(" ", "_").lower()
                filename = f"{folder_name}_{file_safe_label}_{i}.npz"
                save_path = os.path.join(output_root, filename)
                np.savez_compressed(save_path, emg=segment, label=label)
                print(f"✅ Saved: {filename}")

        except Exception as e:
            print(f"❌ Error in {folder_name}: {e}")


# Save data to .npz file

if __name__ == "__main__":
    RAW_DIR = r"path\to\data\folder\raw"
    OUT_DIR = os.path.join(os.path.dirname(RAW_DIR), "emg")
    batch_process_emg_sessions(RAW_DIR, OUT_DIR, window_size=512)



