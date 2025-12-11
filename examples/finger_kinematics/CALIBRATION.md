# EMG Calibration and Normalization

This guide explains how to handle session-to-session EMG variability caused by:
- Electrode placement differences
- Postural changes (elbow position, forearm orientation)
- Co-contraction of stabilizing muscles
- Baseline EMG activity differences

## Quick Start

### 1. Record a Calibration Session

Before each new recording session, record a 10-15 second calibration:

```bash
# Structure:
# - 0-5s: Rest (hands relaxed, natural posture)
# - 5-10s: Maximum Voluntary Contraction (squeeze fist hard)
# - Optional 10-15s: Hold specific pose (e.g., fingers extended)

# Record with RHX software, save as calibration_session1.rhd
```

### 2. Create Calibration File

```bash
python 5_calibrate_and_normalize.py calibrate \
    --root_dir "G:/path/to/project" \
    --file_path "raw/calibration_session1.rhd" \
    --rest_duration 5.0 \
    --mvc_duration 5.0 \
    --verbose
```

This creates `calibration_session1_calibration.npz` containing:
- **Baseline**: Resting EMG activity per channel
- **MVC factors**: Maximum voluntary contraction values per channel

### 3. Predict with Calibration

```bash
python 4_predict.py file \
    --root_dir "G:/path/to/project" \
    --file_path "raw/test_recording.rhd" \
    --calibration_file "raw/calibration_session1_calibration.npz" \
    --label "finger_sweep" \
    --verbose
```

## How It Works

### Baseline Subtraction

Removes session-specific resting activity:

```
EMG_corrected = EMG_raw - Baseline
```

**Why this helps:**
- Different elbow positions change resting muscle activation
- Forearm orientation affects which muscles stabilize the hand
- Electrode gel impedance varies session-to-session

### MVC Normalization

Normalizes to percentage of maximum:

```
EMG_normalized = EMG_corrected / MVC_factor
```

**Why this helps:**
- Accounts for electrode placement variability
- Normalizes across different muscle activation capacities
- Makes predictions more interpretable (0-100% of max)

## Dataset-Level Normalization

For training, you can also normalize entire datasets:

### Z-Score Normalization

Standardizes features to zero mean, unit variance:

```bash
python 5_calibrate_and_normalize.py normalize \
    --dataset_path "dataset/finger_sweep_kinematics_dataset.npz" \
    --method zscore \
    --verbose
```

This helps with:
- Faster neural network training
- Better generalization across different feature magnitudes
- Numerical stability

## Recommended Workflow

### For Training Data Collection

1. **Record calibration** at start of each session
2. **Record training data** (multiple trials)
3. **Build dataset** including all sessions
4. **Optional**: Normalize dataset with z-score
5. **Train model**

### For Testing/Deployment

1. **Record calibration** at start of session
2. **Create calibration file**
3. **Record test data**
4. **Predict with calibration file**

## Advanced: Multi-Session Robustness

If you have multiple training sessions with calibrations:

1. Apply calibration to each training recording BEFORE building dataset
2. This creates a "calibration-normalized" training set
3. Model learns relationships independent of session-specific baseline

**Implementation** (requires modifying `2_build_dataset.py`):

```python
# In preprocessing loop:
if calibration_file:
    calib = np.load(calibration_file)
    emg_data = (emg_data - calib['baseline'][:, None]) / calib['mvc_factors'][:, None]
```

## Troubleshooting

### Predictions still poor after calibration?

- **Check posture**: Ensure consistent hand/arm position across sessions
- **Electrode placement**: Mark electrode positions for reproducibility
- **MVC quality**: Ensure sufficient muscle activation during MVC recording
- **Timing**: Record calibration immediately before experimental data

### Negative EMG values after baseline subtraction?

- Normal! Bandpass-filtered EMG oscillates around zero
- The envelope extraction later makes values positive
- If concerned, use `np.abs()` after baseline subtraction

### Different number of channels?

- Calibration must match model training channel count
- Use same `--channels` or `--channel_map` for calibration and prediction

##Examples

See complete examples in the `finger_kinematics/` directory:
- `1_extract_joint_angles.py` - Extract labels from video
- `2_build_dataset.py` - Build training dataset
- `3_train_model.py` - Train regression model
- `4_predict.py` - Predict with optional calibration
- `5_calibrate_and_normalize.py` - Calibration utilities (this file)
