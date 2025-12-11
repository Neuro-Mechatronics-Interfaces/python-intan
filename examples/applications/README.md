# GUI Applications

PyQt5-based graphical interfaces for EMG data visualization, dataset building, and real-time prediction pipelines.

## Applications

### 1. Dataset Builder GUI 🆕
**File:** `dataset_builder_gui.py`

Comprehensive interface for building EMG gesture classification datasets with full parameter control.

**Features:**
- Visual file discovery (data + event files)
- Real-time pipeline visualization (7-step processing)
- Support for RHD, NPZ, CSV, and Poly5 formats
- Pre-configured channel mappings (8-8-L, 8-8-R, 16-4)
- Feature selection (MAV, RMS, VAR, WL, ZC, SSC, IEMG)
- Profile save/load for reproducibility
- Command generation and terminal execution

**Launch:**
```bash
python dataset_builder_gui.py
```

**Documentation:** [DATASET_BUILDER_GUI.md](DATASET_BUILDER_GUI.md)

---

### 2. Gesture Pipeline GUI
**File:** `gesture_pipeline_gui.py`

Complete workflow GUI for gesture classification: dataset building → training → real-time prediction.

**Features:**
- Integrated dataset building, model training, and live prediction
- LSL (Lab Streaming Layer) integration for real-time streaming
- Live classification visualization with probability plots
- Profile-based configuration management
- Device control and recording management

**Launch:**
```bash
python gesture_pipeline_gui.py
```

**Configuration:** Edit `gesture_pipeline_profile.json` to customize paths and parameters.

---

### 3. EMG Viewer
**File:** `run_emg_viewer.py`

Real-time EMG signal visualization application.

**Features:**
- Multi-channel waveform display
- Configurable time windows and scaling
- Real-time updates from Intan RHX devices or LSL streams
- Export capabilities

**Launch:**
```bash
python run_emg_viewer.py
```

This is a launcher for `intan.applications.launch_emg_viewer()`.

---

### 4. Trial Selector
**File:** `run_trial_selector.py`

Interactive tool for manual trial segmentation and labeling from EMG recordings.

**Features:**
- Load and visualize EMG recordings
- Manual trial boundary marking
- Label assignment per trial
- Export segmented data and event files

**Launch:**
```bash
python run_trial_selector.py
```

This is a launcher for `intan.applications.launch_emg_trial_selector()`.

---

## Installation

All applications require PyQt5:

```bash
pip install python-intan[gui]
```

Or install manually:
```bash
pip install PyQt5 pyqtgraph
```

## Usage Workflows

### Workflow 1: Build Dataset with GUI
```bash
# 1. Launch dataset builder
python dataset_builder_gui.py

# 2. Select root directory
# 3. Choose file type (rhd/npz/csv/poly5)
# 4. Click "Discover Files"
# 5. Configure parameters (or load profile)
# 6. Click "Build Dataset"
```

### Workflow 2: Complete Gesture Pipeline
```bash
# 1. Edit gesture_pipeline_profile.json with your paths
# 2. Launch pipeline GUI
python gesture_pipeline_gui.py

# 3. Build dataset → Train model → Real-time prediction
#    All from one interface!
```

### Workflow 3: Real-time Visualization
```bash
# 1. Connect Intan device and start RHX software
# 2. Launch EMG viewer
python run_emg_viewer.py

# 3. Configure stream source and visualization settings
```

## Configuration Files

### Dataset Builder Profiles
Saved as JSON in the current directory or specified location:
```json
{
  "root_dir": "/path/to/data",
  "file_type": "rhd",
  "multi_file": true,
  "window_ms": 200,
  "step_ms": 50,
  "features": ["mean_absolute_value", "root_mean_square"],
  "channel_map": "8-8-L"
}
```

### Gesture Pipeline Profile
Edit `gesture_pipeline_profile.json`:
```json
{
  "python": "/path/to/python",
  "root_dir": "/path/to/data",
  "label": "gestures",
  "scripts": {
    "build": "path/to/build_script.py",
    "train": "path/to/train_script.py",
    "realtime": "path/to/predict_script.py"
  }
}
```

## File Organization

```
examples/applications/
├── README.md                         # This file
├── dataset_builder_gui.py            # Dataset building interface
├── DATASET_BUILDER_GUI.md            # Detailed user guide
├── gesture_pipeline_gui.py           # Complete pipeline GUI
├── gesture_pipeline_profile.json     # Configuration template
├── run_emg_viewer.py                 # EMG visualization launcher
└── run_trial_selector.py             # Trial segmentation launcher
```

## Tips

### Dataset Builder
- Use **"Discover Files"** after selecting root directory to preview found files
- **Blue panel** = data files, **Yellow panel** = event files
- Pipeline display updates in real-time as you change settings
- Save profiles for consistent processing across sessions

### Gesture Pipeline
- Edit the profile JSON before first launch to set your Python path and script locations
- Use LSL for real-time streaming if working with multiple data sources
- Profile changes are saved automatically

### EMG Viewer & Trial Selector
- These launch applications from the `intan.applications` module
- For development/customization, see `intan/applications/` in the package source

## Related Examples

- **Gesture Classification:** `examples/gesture_classifier/`
- **Finger Kinematics:** `examples/finger_kinematics/`
- **RHX Device Streaming:** `examples/RHXDevice/`
- **LSL Integration:** `examples/LSL/`

## Troubleshooting

### "ModuleNotFoundError: No module named 'PyQt5'"
Install GUI dependencies:
```bash
pip install python-intan[gui]
```

### Dataset Builder shows no event files
- Ensure event files are in `events/` subdirectory or contain "event" in filename
- Supported patterns: `*_events.txt`, `*_event.txt`, `*.events`, `*.event`
- Any `.txt` file with "event" in the name (case-insensitive)

### Gesture Pipeline can't find scripts
Edit `gesture_pipeline_profile.json` with absolute paths to your script locations.

### EMG Viewer not connecting
- Ensure Intan RHX software is running with TCP server enabled
- Check TCP settings: Command Port 5000, Data Port 5001

## Support

- GitHub Issues: https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues
- Documentation: https://python-intan.readthedocs.io/
- Package Source: `intan/applications/`
