# EMG Gesture Classification Pipeline

Complete end-to-end pipeline for training and deploying EMG gesture classifiers using the `intan` package.

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Build       │     │  2. Train       │     │  3. Predict     │
│  Dataset        │ ──▶ │  Model          │ ──▶ │  (4 modes)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
   .rhd + .event        .npz → .pth          file/batch/record/stream
```

## Quick Start

### Step 1: Build Training Dataset

```bash
# Interactive mode (prompts for all options)
python 1_build_dataset.py

# Or with command-line arguments
python 1_build_dataset.py \
    --root_dir /path/to/experiment \
    --multi_file \
    --window_ms 200 \
    --step_ms 50 \
    --overwrite
```

**Expected directory structure:**
```
experiment/
├── raw/
│   ├── gesture1_251203_195153/
│   │   └── gesture1_251203_195153.rhd
│   └── gesture2_251203_202556/
│       └── gesture2_251203_202556.rhd
├── events/
│   ├── gesture1_emg.event
│   └── gesture2_emg.event
└── training_dataset.npz    # OUTPUT: combined features + labels
```

**Features:**
- ✅ **Multi-file support**: Automatically discovers and processes multiple recordings
- ✅ **Exclude patterns**: Filter out specific gestures during dataset building
- ✅ **Interactive prompts**: GUI file picker + terminal text inputs
- ✅ **Config persistence**: Saves settings to `.gesture_config` for reuse
- ✅ **Error reporting**: Detailed logs for missing event files

### Step 2: Train the Model

```bash
# Interactive mode
python 2_train_model.py

# Or with arguments
python 2_train_model.py \
    --root_dir /path/to/experiment \
    --overwrite
```

**Features:**
- ✅ **PyTorch CNN classifier** with PCA dimensionality reduction
- ✅ **Early stopping** to prevent overfitting
- ✅ **Train/val split** with stratification
- ✅ **Automatic checkpointing** of best model

**Output files:**
```
experiment/
└── model/
    ├── model.pth           # Trained PyTorch weights
    ├── scaler.pkl          # Feature normalization
    ├── pca.pkl             # PCA dimensionality reduction
    ├── label_encoder.pkl   # Class labels
    ├── metadata.json       # Pipeline parameters
    └── metrics.json        # Classification report
```

### Step 3: Run Predictions

The unified prediction CLI supports **4 modes**:

```bash
# Interactive mode selection
python 3_predict.py

# Or specify mode directly:
python 3_predict.py file --file_path recording.rhd --events_file labels.event
python 3_predict.py batch --rhd_glob "raw/**/*.rhd" --events_dir events/
python 3_predict.py record --seconds 10
python 3_predict.py stream --infer_hz 20 --use_lsl
```

#### Mode 1: File (Offline Single File)

Predict from a single RHD file with optional event comparison:

```bash
python 3_predict.py file \
    --file_path /path/to/recording.rhd \
    --events_file /path/to/labels.event \
    --verbose
```

**Output:**
- `predictions/{filename}_predictions.txt` - Timestamped predictions
- `predictions/{filename}_evaluation.json` - Metrics if events file provided

#### Mode 2: Batch (Multiple Files)

Process multiple files and generate aggregated metrics:

```bash
python 3_predict.py batch \
    --rhd_glob "raw/**/*.rhd" \
    --events_dir events/ \
    --save_eval
```

**Output:** Aggregated classification report across all files

#### Mode 3: Record (Device Recording)

Record from Intan device for fixed duration and predict:

```bash
python 3_predict.py record \
    --seconds 10 \
    --event_file /path/to/labels.event
```

#### Mode 4: Stream (Real-time)

Real-time streaming prediction from device:

```bash
python 3_predict.py stream \
    --infer_hz 20 \
    --smooth_k 5 \
    --use_lsl \
    --seconds_total 60
```

**Options:**
- `--infer_hz`: Prediction rate (Hz)
- `--smooth_k`: Majority vote window size for smoothing
- `--use_lsl`: Publish predictions to Lab Streaming Layer
- `--seconds_total`: Duration (None = infinite)

## Feature Specification

The unified dataset builder supports flexible feature extraction:

**Default Features:**
- **Time Domain**: RMS, MAV, VAR, WL, ZC, SSC, MAX, MIN, RANGE
- **Total**: ~896 features (128 channels × 7 base features)

**Supported Input Formats:**
- **RHD files**: Native Intan recordings
- **NPZ files**: Pre-processed numpy archives
- **DAT files**: Raw binary data
- **CSV files**: Third-party exports

The builder auto-detects format and handles single or multiple files via interactive prompts or glob patterns

## Event File Format

Label files (`.event` or `.txt`) use a simple format:
```
# timestamp_samples  label
0       Start
5000    WristFlexion
25000   Rest
45000   WristExtension
65000   Rest
...
```

Timestamps are in samples (not seconds). The label applies from that sample until the next timestamp.

## Configuration Persistence

All scripts use `.gesture_config` to save/load parameters:

```json
{
    "root_dir": "/path/to/experiment",
    "window_ms": 200,
    "step_ms": 50,
    "feature_spec": ["rms", "mav", "var", "wl", "zc", "ssc"],
    "use_pca": true,
    "pca_n_components": 30,
    "ignore_labels": ["Start", "End", "None", "Unknown"]
}
```

**Incremental saving** ensures consistency across the pipeline:
1. Dataset builder saves window parameters
2. Trainer adds model hyperparameters
3. Predictor inherits all settings automatically

You can override any parameter via command-line arguments.

## GPU Acceleration

Training benefits significantly from GPU acceleration:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### WSL2 + CUDA Setup (Windows)

1. Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
2. Install [NVIDIA CUDA drivers for WSL](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0)
3. Verify with `nvidia-smi`

## Troubleshooting

### "Config file not found"
First run of any script creates `.gesture_config`. Ensure `--root_dir` points to your experiment directory.

### "Dataset file not found"
Run `1_build_dataset.py` first. Check for `dataset/training_dataset.npz` in your root directory.

### "Feature dimension mismatch"
Prediction channels must match training. Verify:
- Same number of channels enabled
- Same feature specification (`feature_spec` in config)
- PCA settings consistent (check `use_pca` flag)

### "No events found" (Batch Mode)
Event files must be in `<root_dir>/events/` with naming `<recording_basename>.event` (matching the RHD file stem).

### "Model not converging"
Try adjusting:
- `--learning_rate` (default 0.001)
- `--batch_size` (default 64)
- `--patience` for early stopping (default 10 epochs)
- Enable `--use_pca` to reduce dimensionality

## Advanced Topics

### Custom Model Architecture

The default `EMGClassifier` is a PyTorch CNN with dropout:
```python
Input (n_features) → Dense(256) → ReLU → Dropout(0.3)
                   → Dense(128) → ReLU → Dropout(0.3)
                   → Dense(n_classes) → Softmax
```

To use a custom architecture:
```python
from intan.ml import ModelManager
from torch import nn

class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(...)
    
    def forward(self, x):
        return self.layers(x)

manager = ModelManager(model_cls=CustomModel)
manager.train(X_train, y_train, ...)
```

### Real-time LSL Integration

Stream predictions to Lab Streaming Layer for multi-system synchronization:
```bash
python 3_predict.py stream --use_lsl --infer_hz 20
```

Subscribe in another application:
```python
from intan.interface import LSLSubscriber
sub = LSLSubscriber(stream_name="EMG_Predictions")
prediction = sub.pull_sample()
```

### Evaluation Metrics

Prediction modes support automatic evaluation when event files are provided:
- **File mode**: Generates `_evaluation.json` with per-class F1-scores
- **Batch mode**: Aggregates metrics across all files
- **Output format**: sklearn classification report (precision/recall/F1/support)

Example output:
```json
{
    "HandOpen": {"precision": 0.92, "recall": 0.94, "f1-score": 0.93, "support": 145},
    "Rest": {"precision": 0.97, "recall": 0.98, "f1-score": 0.97, "support": 567},
    "accuracy": 0.96
}
```

## Citation

If you use this pipeline in your research, please cite:
```
@software{intan_python,
    author = {Shulgach, Jonathan},
    title = {Intan Python: EMG Processing and Classification},
    url = {https://github.com/jshulgach/intan-python}
}
```
