# Dataset Builder GUI - User Guide

## Overview

The Enhanced Dataset Builder GUI provides a comprehensive interface for configuring and building EMG gesture classification datasets. It offers full control over all processing parameters with visual feedback of the feature extraction pipeline.

## Features

### 📁 Project Setup
- **Root Directory Selection**: Browse and select your project root containing EMG recordings and event files
- **Dataset Label**: Optional prefix for output filenames
- **Auto-Discovery**: Automatically finds all data and event files in your project

### 📄 File Settings
- **File Type Support**: RHD, NPZ, or CSV formats
- **Multi-File Mode**: Aggregate multiple recordings into a single dataset
- **Pattern Filtering**: 
  - Include pattern (e.g., `train_*` to only process training files)
  - Exclude pattern (e.g., `*_test` to skip test files)
- **File Discovery**: One-click discovery shows all found data and event files

### 📡 Channel Configuration
- **Channel Mapping**: Pre-configured mappings (8-8-L, 8-8-R, 16-4) for common electrode arrays
- **Manual Selection**: Specify exact channels (e.g., `0:64` or `0,5,10-20`)
- **Non-Strict Mapping**: Allow missing channels in mapping (useful for damaged electrodes)
- **Orientation Remap**: Spatial transforms for rotated electrodes (mirror, rotate90)

### ⚙️ Signal Processing
- **Paper-Style Mode**: One-click preset (120Hz highpass, RMS only, 250ms non-overlapping windows)
- **Custom Windows**: Configure window size (50-1000ms) and step size (10-500ms)
- **Overlap Display**: Real-time calculation of window overlap percentage
- **Modality Selection**: EMG only, IMU only, or both combined
- **IMU Options**: Feature mode (mean/rich) and normalization (zscore/robust)

### 🔧 Feature Selection
Select which features to extract from each EMG channel:
- Mean Absolute Value (MAV)
- Root Mean Square (RMS)
- Variance
- Waveform Length
- Zero Crossings
- Slope Sign Changes
- Integrated EMG

**Bulk Controls**: "Select All" / "Select None" buttons for quick configuration

### 🏷️ Label Filtering
- **Ignore Labels**: Comma-separated list of labels to exclude (e.g., "Start, End, None")
- **Case-Insensitive**: Toggle for case-insensitive label matching
- **Keep Trial Numbers**: Optionally preserve trial numbers in labels (e.g., `fist_3` vs `fist`)

### 🔬 Advanced Options
- **Overwrite**: Force overwrite of existing output files
- **Verbose Logging**: Enable detailed logging for debugging
- **Config File**: Load all settings from a JSON configuration file

## Pipeline Visualization

The right panel displays a **real-time visualization** of your feature extraction pipeline:

```
📊 Feature Extraction Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load Data
   └─ Format: RHD files
   └─ Mode: Multi-file aggregation

2. Channel Selection
   └─ Mapping: 8-8-L

3. Signal Preprocessing
   └─ Bandpass: 20-500 Hz
   └─ Notch: 60 Hz

4. Sliding Window
   └─ Window: 200 ms
   └─ Step: 50 ms (75% overlap)

5. Feature Extraction
   └─ Features per channel:
      • mean_absolute_value
      • root_mean_square
      • variance
   └─ Total features/window: 3 × n_channels

6. Label Matching
   └─ Source: Event files (.txt)
   └─ Ignored labels: Start, End, None

7. Save Dataset
   └─ Format: NPZ (NumPy compressed)
   └─ Contains: X (features), y (labels), metadata
```

This updates **automatically** as you change settings!

## File Discovery Panel

Shows all files found in your project:
- **Data Files**: All matching files of selected type (RHD/NPZ/CSV)
- **Event Files**: All `*_events.txt` files for label extraction

Count displayed in status bar: "Found 12 RHD files, 12 event files"

## Usage Workflow

### 1. Quick Start (Default Settings)
1. Click **"Browse..."** and select your project root directory
2. Click **"🔄 Discover Files"** to find all data/event files
3. Review the pipeline visualization (defaults are usually good!)
4. Click **"🚀 Build Dataset"**

### 2. Custom Configuration
1. Select root directory
2. **File Settings**:
   - Choose file type (RHD/NPZ/CSV)
   - Enable/disable multi-file mode
   - Add include/exclude patterns if needed
3. **Channel Configuration**:
   - Select channel mapping or specify manually
   - Enable orientation remap if electrodes were rotated
4. **Signal Processing**:
   - Adjust window/step sizes for your application
   - OR enable paper-style for standard preprocessing
5. **Feature Selection**:
   - Check/uncheck features as needed
   - Or use "Select All"/"Select None"
6. **Label Filtering**:
   - Add labels to ignore (e.g., rest periods, calibration)
7. Click **"🚀 Build Dataset"**

### 3. Load from Profile
1. Click **"📂 Load Profile"**
2. Select a previously saved `.json` configuration
3. All settings will be restored
4. Click **"🚀 Build Dataset"**

### 4. Save Your Configuration
1. Configure all settings as desired
2. Click **"💾 Save Profile"**
3. Save as `.json` for future use

## Common Scenarios

### Scenario 1: Single Recording with Default Settings
```
Root: /data/my_project/
File Type: rhd
Multi-file: ❌ (unchecked)
Channels: (blank - use all)
Window: 200ms, Step: 50ms
Features: All selected
```

### Scenario 2: Multiple Training Recordings
```
Root: /data/my_project/
File Type: rhd
Multi-file: ✅ (checked)
Include Pattern: train_*
Exclude Pattern: *_test
Channels: 0:64
Window: 200ms, Step: 50ms
Features: RMS, MAV, Variance
Ignore Labels: Start, End, Rest, None
```

### Scenario 3: Paper-Style Preprocessing
```
Root: /data/my_project/
File Type: rhd
Multi-file: ✅
Paper Style: ✅ (checked) ← This sets everything automatically!
  → Window: 250ms (auto-set)
  → Step: 250ms (auto-set)
  → Features: RMS only (recommended)
  → Preprocessing: 120Hz highpass
```

### Scenario 4: HD-EMG with Custom Mapping
```
Root: /data/hdemg_project/
File Type: rhd
Multi-file: ✅
Channel Map: 8-8-L ← For 64-channel grid
Orientation Remap: mirror ← If electrodes were flipped
Channels: (blank - uses mapping)
Features: All
```

### Scenario 5: EMG + IMU Combined
```
Root: /data/multimodal/
File Type: csv
Multi-file: ✅
Modality: both ← EMG + IMU
IMU Features: rich ← More IMU features
IMU Norm: zscore
Features: All EMG features
```

## Command Generation

When you click **"🚀 Build Dataset"**, the GUI:
1. Validates all settings
2. Generates the complete command-line call
3. Shows you the command in a preview dialog
4. Launches it in a new terminal window

Example generated command:
```bash
python 1_build_dataset.py \
    --root_dir /data/my_project \
    --file_type rhd \
    --multi_file \
    --channels 0:64 \
    --channel_map 8-8-L \
    --window_ms 200 \
    --step_ms 50 \
    --ignore_labels Start End None \
    --ignore_case \
    --overwrite \
    --verbose
```

You can copy this command to run it manually or in scripts!

## Tips & Best Practices

### ✅ Do's
- **Always discover files first** to verify the GUI found everything
- **Check the pipeline visualization** before building
- **Save profiles** for repeated experiments
- **Use paper-style** for reproducible research
- **Enable verbose** when troubleshooting
- **Review ignore labels** - "Rest" might be a valid class!

### ❌ Don'ts
- Don't forget to create event files before building
- Don't use too small window sizes (<50ms typically unstable)
- Don't select zero features (GUI will warn you)
- Don't mix different electrode orientations without remapping

## Troubleshooting

### No files discovered
- Check root directory path is correct
- Verify file type matches your data (RHD vs NPZ vs CSV)
- Make sure files aren't in excluded patterns

### No event files found
- Event files must end with `_events.txt` or `.events`
- They should be in the same directory (or subdirectory) as data files
- Check the event list widget to see what was found

### Build fails
- Enable verbose logging for detailed error messages
- Check that all recordings have corresponding event files
- Verify channel selections are valid for your data
- Review the generated command in the preview dialog

### Features look wrong
- Check the pipeline visualization matches your intent
- Verify window/step sizes are appropriate for your gestures
- Ensure correct features are selected for your task

## Keyboard Shortcuts

- **Ctrl+O**: Browse root directory
- **Ctrl+S**: Save profile
- **Ctrl+L**: Load profile
- **F5**: Discover files
- **Ctrl+Enter**: Build dataset (if validated)

## Output

The dataset will be saved in `root_dir/dataset/` as:
- `{label}_dataset.npz` (if label provided)
- `dataset.npz` (if no label)

Contains:
- `X`: Feature matrix (n_samples, n_features)
- `y`: Label array (n_samples,)
- `y_id`: Integer label IDs
- `class_names`: Unique class names
- `metadata`: All processing parameters

## Integration with Training

After building, use the dataset for training:
```bash
python 2_train_model.py --root_dir /data/my_project
```

The training script will automatically find and load the dataset!
