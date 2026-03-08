# Finger Kinematics Pipeline - Complete Workflow

Complete parameter reference for all 8 pipeline scripts. For quick start instructions, see [README.md](README.md).

## Essential Scripts (1-6)

### 1️⃣ `1_synchronize_emg_video.py` - Temporal Alignment

**Purpose:** Find time offsets between EMG and video recordings.

**⚠️ CRITICAL:** Run this before Step 4 (dataset building). Observed offsets up to 9.7 seconds!

```bash
# Batch process (RECOMMENDED)
python 1_synchronize_emg_video.py batch --root_dir /data --verbose

# Single pair
python 1_synchronize_emg_video.py find \
    --root_dir /data \
    --emg_file raw/recording.rhd \
    --landmarks_file media/landmarks/recording_landmarks.npz

# Verify alignment
python 1_synchronize_emg_video.py plot \
    --root_dir /data \
    --emg_file raw/recording.rhd \
    --landmarks_file media/landmarks/recording_landmarks.npz \
    --offset_file sync/recording_sync.json
```

**Output:** `sync/*.json` files with offsets and confidence scores (>0.3 = good)

---

### 2️⃣ `2_extract_joint_angles.py` - Video → Joint Angles

**Commands:**
```bash
# Interactive (file dialog)
python 2_extract_joint_angles.py

# Specified path
python 2_extract_joint_angles.py --video_path recording.mp4 --visualize
```

**Output:**
- `media/landmarks/*_landmarks.npz` - 21 3D hand landmarks
- `media/landmarks/*_angles.csv` - 5 joint angles (thumb→pinky)

**Skip if:** You already have joint angle CSV files from other sources

---

### 3️⃣ `3_calibrate_and_normalize.py` - EMG Normalization (Optional)

**Purpose:** Create calibration files for session-to-session EMG normalization.

**Why before dataset building?** If you normalize training data, the model expects normalized data at inference too. Calibration should happen BEFORE building the dataset.

**Commands:**
```bash
# Create calibration file from rest + MVC recording
python 3_calibrate_and_normalize.py \
    --root_dir /data \
    --file_path calibration_recording.rhd \
    --rest_duration 5 \
    --mvc_duration 5

# Normalize existing dataset
python 3_calibrate_and_normalize.py \
    --root_dir /data \
    --calibration_file calibration.npz \
    --input_dataset raw_dataset.npz \
    --output_dataset normalized_dataset.npz
```

**Output:** `calibration/*.npz` files with baseline/MVC normalization factors

**Skip if:** You want to train on raw EMG without normalization

---

### 4️⃣ `4_build_dataset.py` - EMG Features + Aligned Labels

**Purpose:** Create training dataset with synchronized EMG features and joint angle labels.

**Key:** Automatically loads sync offsets from `sync/` directory!

**Commands:**
```bash
# Multi-file (RECOMMENDED)
python 4_build_dataset.py --root_dir /data --multi_file --verbose

# Single file
python 4_build_dataset.py \
    --root_dir /data \
    --file_path raw/recording.rhd \
    --angles_file media/landmarks/recording_angles.csv

# Custom windowing
python 4_build_dataset.py \
    --root_dir /data \
    --multi_file \
    --window_ms 250 \
    --step_ms 50 \
    --channels 0:64
```

**Output:** `dataset/*_kinematics_dataset.npz`

**What it does:**
1. Loads EMG data
2. Applies bandpass filter (20-450 Hz)
3. Extracts features (RMS, MAV, VAR, etc.) in sliding windows
4. Loads joint angle CSVs
5. **Applies sync offset: `angle_timestamps -= offset_sec`**
6. Interpolates angles to EMG window timestamps
7. Saves aligned dataset

---

### 4️⃣ `5_train_model.py` - Train Regression Model

**Purpose:** Train EMGRegressor to predict joint angles from EMG features.

**Commands:**
```bash
# Basic training
python 5_train_model.py --root_dir /data --epochs 3000 --patience 50

# With PCA dimensionality reduction
python 5_train_model.py \
    --root_dir /data \
    --use_pca \
    --pca_variance 0.95 \
    --epochs 3000

# Resume from checkpoint
python 5_train_model.py --root_dir /data --resume
```

**Output:** `model/` directory with trained model

**Metrics:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)  
- R² (coefficient of determination)

**Architecture:**
```
Input → BatchNorm → Dense(256) → ReLU → Dropout(0.2)
      → BatchNorm → Dense(128) → ReLU → Dropout(0.2)
      → Dense(n_joints)
```

---

### 6️⃣ `6_predict.py` - Inference and Evaluation

**Purpose:** Predict joint angles from new EMG recordings.

**Key:** Automatically applies sync offsets when comparing to ground truth!

**Commands:**
```bash
# Single file with ground truth
python 6_predict.py file \
    --root_dir /data \
    --file_path raw/test.rhd \
    --angles_file media/landmarks/test_angles.csv \
    --plot

# With temporal smoothing (reduces noise)
python 6_predict.py file \
    --root_dir /data \
    --file_path raw/test.rhd \
    --angles_file media/landmarks/test_angles.csv \
    --smooth \
    --smooth_method savgol \
    --plot

# Batch evaluation
python 6_predict.py batch \
    --root_dir /data \
    --rhd_glob "raw/test_*.rhd" \
    --angles_dir media/landmarks/

# Real-time device recording (coming soon)
python 6_predict.py record --seconds 10

# Real-time streaming (coming soon)
python 6_predict.py stream --infer_hz 20
```

**Smoothing Methods:**
- `savgol` - Savitzky-Golay (default, edge-preserving)
- `lowpass` - Butterworth lowpass filter
- `gaussian` - Gaussian kernel smoothing
- `exponential` - Exponential moving average (real-time compatible)

**Output:**
- `*_predictions.txt` - Timestamped predictions
- `*_evaluation.json` - Metrics (if ground truth provided)
- Optional plot

---

### 7️⃣ `7_analyze_predictions.py` - Comprehensive Diagnostics (OPTIONAL)

**Purpose:** In-depth analysis of prediction quality with statistical diagnostics.

**Commands:**
```bash
# Basic analysis with plots
python 7_analyze_predictions.py \
    --pred_file predictions.txt \
    --angles_file angles.csv \
    --joint_names Thumb Index Middle Ring Pinky

# Save comprehensive report
python 7_analyze_predictions.py \
    --pred_file predictions.txt \
    --angles_file angles.csv \
    --output_dir diagnostics/ \
    --fs 20.0
```

**Generates:**
- Correlation plots (per-joint scatter with R²)
- Bland-Altman plots (agreement analysis)
- Error distributions and Q-Q plots
- Cross-correlation matrices between joints
- Time-series error evolution
- Frequency analysis of residuals

---

### 8️⃣ `8_render_prediction_video.py` - Video Overlay (OPTIONAL)

**Purpose:** Create annotated video with prediction overlay for visual validation.

**Commands:**
```bash
# Basic skeleton overlay
python 8_render_prediction_video.py \
    --video test.mp4 \
    --pred predictions.txt \
    --landmarks landmarks.npz

# With error visualization
python 8_render_prediction_video.py \
    --video test.mp4 \
    --pred predictions.txt \
    --landmarks landmarks.npz \
    --angles angles.csv \
    --show_errors \
    --error_threshold 15

# Disable time-series plots (faster rendering)
python 8_render_prediction_video.py \
    --video test.mp4 \
    --pred predictions.txt \
    --landmarks landmarks.npz \
    --no_plots
```

**Features:**
- Hand skeleton overlay with joint angles
- Color-coded error visualization (green→yellow→red)
- Real-time angle plots at bottom
- Frame-by-frame error statistics
- Synchronized video playback

---

### ❌ DEPRECATED: `6_calibrate_and_normalize.py` - Session Normalization

**This script was moved to step 3 in the refactoring. See `3_calibrate_and_normalize.py` above.**

---

## Typical Workflow

### Initial Setup (Once per dataset)

```bash
# 1. Synchronize all recordings (CRITICAL!)
python 1_synchronize_emg_video.py batch --root_dir /data --verbose

# 2. Extract joint angles from videos
for video in videos/*.mp4; do
    python 2_extract_joint_angles.py --video_path "$video"
done

# 3. Build training dataset (sync applied automatically)
python 4_build_dataset.py --root_dir /data --multi_file --label training

# 4. Train model
python 5_train_model.py --root_dir /data --label training --epochs 3000
```

### Testing New Recording

```bash
# 1. Synchronize new recording with its video
python 1_synchronize_emg_video.py find \
    --root_dir /data \
    --emg_file raw/test_new.rhd \
    --landmarks_file media/landmarks/test_new_landmarks.npz

# 2. Predict (sync applied automatically)
python 6_predict.py file \
    --root_dir /data \
    --label training \
    --file_path raw/test_new.rhd \
    --angles_file media/landmarks/test_new_angles.csv \
    --smooth \
    --plot

# 3. Analyze predictions (optional)
python 7_analyze_predictions.py \
    --pred_file test_new_predictions.txt \
    --angles_file media/landmarks/test_new_angles.csv \
    --output_dir diagnostics/test_new/ \
    --fs 20.0

# 4. Render annotated video (optional)
python 8_render_prediction_video.py \
    --video videos/test_new.mp4 \
    --pred test_new_predictions.txt \
    --landmarks media/landmarks/test_new_landmarks.npz \
    --angles media/landmarks/test_new_angles.csv \
    --show_errors
```

---

## Key Synchronization Points

### How sync offsets are used:

**Step 1 (Synchronize):**
- Computes: `offset_sec` via cross-correlation
- Saves: `sync/*.json`

**Step 3 (Build Dataset):**
```python
offset_sec = load_sync_offset(sync_dir, file_stem)
# Subtract because positive offset means landmarks are delayed
angle_timestamps_aligned = angle_timestamps - offset_sec
# Then interpolate to EMG window timestamps
```

**Step 5 (Predict):**
```python
offset_sec = load_sync_offset(sync_dir, file_stem)
# Apply to ground truth before comparison
angle_timestamps_aligned = angle_timestamps - offset_sec
# Compute metrics on aligned data
```

---

## Critical Rules

1. **Always run Step 1 before Step 3**
   - Dataset building requires sync offsets
   - Missing sync files = 0.0 offset assumed (likely wrong!)

2. **Sync files must match recording names**
   - `train_finger_sweep_251110_125854.rhd` → `train_finger_sweep_251110_125854_sync.json`
   - Or stripped: `train_finger_sweep_sync.json`

3. **Offset sign convention**
   - Positive offset = landmarks DELAYED relative to EMG
   - To align: **SUBTRACT** offset from landmark timestamps
   - `aligned_time = landmark_time - offset_sec`

4. **Confidence threshold**
   - >0.3: Good alignment
   - <0.3: Manual verification recommended
   - Check with: `python 1_synchronize_emg_video.py plot ...`

---

## Troubleshooting

### "No sync file found"
→ Run Step 1 for that recording

### "Synchronization confidence low (<0.3)"
→ Verify:
- Both EMG and video contain clear movement
- Landmark extraction quality
- Recording timing wasn't interrupted

### "Model trains well but test set fails (R²<0)"
→ Possible causes:
- Test set not synchronized (run Step 1!)
- Different movement patterns than training
- Overfitting (try more diverse training data)

### "Dataset building says sync offset applied, but performance still bad"
→ Verify sync quality:
```bash
python 1_synchronize_emg_video.py plot \
    --root_dir /data \
    --emg_file raw/recording.rhd \
    --landmarks_file media/landmarks/recording_landmarks.npz \
    --offset_file sync/recording_sync.json \
    --save alignment_check.png
```

---

## See Also

- `SYNCHRONIZATION.md` - Detailed sync algorithm documentation
- `CALIBRATION.md` - Session normalization protocol
- `README.md` - Full pipeline documentation
