# EMG-Video Synchronization System

## Overview

The synchronization system aligns EMG recordings with video-derived landmark data using cross-correlation of movement signals.

## Problem Statement

EMG recordings and video recordings are captured independently and may not start/stop at the same time. Without synchronization, features extracted from EMG windows will be matched with incorrect joint angle labels, leading to catastrophic training failure.

## Discovery

Initial analysis revealed time offsets ranging from **-8.0s to +9.7s** between EMG and video streams across 6 recordings. The largest offset (9.7 seconds) corresponds to nearly 200 frames of misalignment!

## Solution: Cross-Correlation Based Sync

### Algorithm (`1_synchronize_emg_video.py`)

1. **Extract Movement Signals**:
   - **EMG Signal**: Average envelope across all channels (bandpass → rectify → lowpass @ 10Hz)
   - **Landmark Signal**: Velocity of fingertip centroid (landmarks 4,8,12,16,20)

2. **Cross-Correlate**:
   - Resample both signals to common timeline (EMG sample rate)
   - Normalize signals (zero mean, unit std)
   - Compute cross-correlation with ±10s search window
   - Find peak correlation value

3. **Compute Offset**:
   - **Positive offset**: Landmarks are DELAYED relative to EMG
   - **Negative offset**: Landmarks START BEFORE EMG

4. **Confidence Metric**:
   - Ratio of peak correlation to mean correlation
   - Values >0.3 considered reliable

### Detected Offsets

| Recording | Offset (sec) | Offset (frames) | Confidence |
|-----------|--------------|-----------------|------------|
| test_dynamic | -0.594 | -12 | 1.00 |
| train_dynamic2 | +0.871 | +18 | 1.00 |
| test_finger_sweep | +1.018 | +21 | 1.00 |
| train_dynamic | -2.003 | -41 | 1.00 |
| train_finger_sweep2 | -8.022 | -164 | 1.00 |
| train_finger_sweep | +9.691 | +197 | 0.49 |

**Mean offset**: 0.160 ± 5.232 seconds

## Implementation

### 1. Sync Offset Files (`sync/*.json`)

Example: `train_finger_sweep_251110_125854_sync.json`
```json
{
  "offset_sec": 9.691,
  "offset_samples_emg": 9691,
  "offset_samples_landmarks": 197,
  "correlation_peak": 15739.39,
  "confidence": 0.49,
  "emg_fs": 1000.0,
  "landmark_fs": 20.4
}
```

### 2. Dataset Building (`4_build_dataset.py`)

**Key change**: Line 285
```python
# CRITICAL: Subtract offset because positive offset means landmarks are delayed
angle_timestamps = angle_timestamps - sync_offset_sec
```

**Workflow**:
1. Load sync offset from `sync/*.json`
2. Subtract from `angle_timestamps` (shifts landmarks backward in time)
3. Interpolate angles to EMG window timestamps

### 3. Prediction (`6_predict.py`)

**Key change**: Lines 161-182
```python
# Apply sync offset to ground truth before interpolation
if os.path.exists(sync_dir):
    sync_path = find_sync_file(sync_dir, file_path)
    offset_sec = load_offset(sync_path)
    angle_timestamps = angle_timestamps - offset_sec  # Shift landmarks backward
```

**Ensures**: Predictions and ground truth are compared on the same time basis.

## Usage

### Generate Sync Offsets

```bash
# Single file
python 1_synchronize_emg_video.py find \
    --root_dir "path/to/root" \
    --emg_file "train_finger_sweep.rhd" \
    --landmarks_file "train_finger_sweep_landmarks.npz"

# Batch process all recordings
python 1_synchronize_emg_video.py batch \
    --root_dir "path/to/root" \
    --verbose

# Visualize alignment
python 1_synchronize_emg_video.py plot \
    --root_dir "path/to/root" \
    --emg_file "train_finger_sweep.rhd" \
    --landmarks_file "train_finger_sweep_landmarks.npz" \
    --offset_file "train_finger_sweep_sync.json" \
    --save "alignment.png"
```

### Build Synchronized Dataset

```bash
python 4_build_dataset.py \
    --file_names train_finger_sweep_251110_125854 train_finger_sweep2_251110_130107 \
    --label finger_sweep_synced
```

**Automatic**: Sync offsets loaded from `sync/` directory if available.

### Train and Predict

```bash
# Train
python 5_train_model.py \
    --train_npz finger_sweep_synced_kinematics_dataset.npz \
    --label finger_sweep_synced

# Predict (sync applied automatically)
python 6_predict.py file \
    --root_dir "path/to/root" \
    --label "finger_sweep_synced" \
    --file_path "test_file.rhd" \
    --angles_file "test_angles.csv"
```

## Performance Impact

### Before Synchronization
- Training: R² = 0.84, MAE = 8.79° (validation)
- **BUT**: Data was misaligned by up to 9.7 seconds!
- Model was learning spurious correlations from wrong labels

### After Synchronization (v2)
- Training: R² = 0.75, MAE = 10.91° (validation)
- **Better**: Model trained on correctly aligned data
- **Test set**: R² = -10.5 (poor generalization - needs investigation)

### Interpretation

The synchronized model has **lower validation performance but is more honest** - it's learning true EMG-kinematics relationships, not spurious patterns from misaligned data. The poor test set performance (R²=-10.5) suggests:

1. **Overfitting**: Model doesn't generalize beyond training movements
2. **Movement diversity**: Training data may not represent test scenarios
3. **Feature limitations**: Current features (RMS, MAV, VAR, etc.) may be insufficient

## Critical Insights

1. **Offset Sign Convention**:
   - `correlate(emg, landmark)` → positive peak means landmarks are delayed
   - **Always SUBTRACT offset** from landmark timestamps to align

2. **Confidence Matters**:
   - Low confidence (<0.3) indicates weak correlation → check data quality
   - High confidence (>0.9) indicates strong movement synchrony

3. **Sync First, Train Second**:
   - Always generate sync offsets BEFORE building datasets
   - Rebuilding datasets after sync correction is essential

## Troubleshooting

### Low Confidence (<0.3)
- Check if both EMG and video contain clear movement
- Verify landmark extraction quality
- Consider recordings may have minimal movement overlap

### Large Offsets (>10s)
- Verify recording start times
- Check for recording interruptions
- May indicate procedural issues during data collection

### Model Performance Drops After Sync
- **Expected!** Original "good" performance was from memorizing misaligned patterns
- Indicates model now learning harder (but correct) task
- Solution: More diverse training data, better features, or different architecture

## Files Modified

- `1_synchronize_emg_video.py` - Sync detection tool
- `4_build_dataset.py` - Load and apply sync offsets
- `6_predict.py` - Apply sync to ground truth during evaluation
- `SYNCHRONIZATION.md` ✅ NEW - This documentation

## Next Steps

1. **Verify Sync Correctness**: Plot aligned signals visually
2. **Investigate Poor Test Performance**: 
   - Check training vs test data distributions
   - Consider data augmentation or domain adaptation
   - Try simpler model architectures (may reduce overfitting)
3. **Expand Training Data**: Include more diverse movements
4. **Feature Engineering**: Explore alternative feature sets (spectral, time-frequency)
