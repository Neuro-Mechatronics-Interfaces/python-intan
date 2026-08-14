# Paper-Style EMG Gesture Classification Replication

This directory contains scripts to replicate the **exact signal processing and classification approach** described in the Journal of Neural Engineering paper for EMG gesture recognition using high-density surface EMG.

## 📋 Overview

The paper's approach consists of:

1. **Signal Processing**:
   - 120 Hz high-pass filter (4th-order Butterworth)
   - RMS feature extraction
   - 250 ms non-overlapping windows
   - Z-score normalization

2. **Dimensionality Reduction**:
   - PCA to K=30 components

3. **Classification**:
   - Neural network: 30 → 512 → 512 → N classes
   - Dropout 0.2 after each hidden layer
   - No batch normalization
   - Adam optimizer, 200 epochs, batch size 32
   - Categorical cross-entropy loss

## 🚀 Quick Start

### Step 1: Build Training Dataset (Paper Style)

```bash
python 1_build_training_dataset.py \
    --root_dir /path/to/your/data \
    --label paper_replication \
    --overwrite \
    --verbose
```

**What this does:**
- Loads CSV recordings from `<root_dir>/csv/` or `<root_dir>/raw/`
- Applies 120 Hz high-pass filter (exactly as in paper)
- Extracts RMS features over 250 ms non-overlapping windows
- Aligns labels from event files in `<root_dir>/events/`
- Saves dataset as `<root_dir>/paper_replication_training_dataset.npz`

### Step 2: Train Model (Paper Style)

```bash
python 2_train_model.py \
    --root_dir /path/to/your/data \
    --label paper_replication \
    --overwrite \
    --verbose
```

**What this does:**
- Loads the dataset from Step 1
- Applies z-score normalization
- Reduces dimensions using PCA (M channels → 30 components)
- Trains the paper's neural network architecture (30→512→512→N)
- Uses Adam optimizer with 200 epochs and batch size 32
- Saves model artifacts to `<root_dir>/model/`

## 📁 File Structure

Your data directory should be organized as:

```
your_data/
├── csv/ (or raw/)
│   ├── recording1.csv
│   ├── recording2.csv
│   └── ...
├── events/
│   ├── recording1_emg.event
│   ├── recording2_emg.event
│   └── ...
└── model/  (created automatically)
    ├── paper_replication_model.pth
    ├── paper_replication_scaler.pkl
    ├── paper_replication_pca.pkl
    ├── paper_replication_label_encoder.pkl
    ├── paper_replication_metadata.json
    └── paper_replication_metrics.json
```

## ⚙️ Advanced Options

### Dataset Building

```bash
# Use specific recordings only
python 1_build_training_dataset.py \
    --root_dir /path/to/data \
    --label my_experiment \
    --csv_names recording1.csv recording3.csv \
    --verbose

# Select specific channels (e.g., first 32 channels)
python 1_build_training_dataset.py \
    --root_dir /path/to/data \
    --label subset_32ch \
    --channels 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
    --overwrite

# Custom save location
python 1_build_training_dataset.py \
    --root_dir /path/to/data \
    --label experiment \
    --save_path /custom/path/dataset.npz
```

### Model Training

```bash
# Custom hyperparameters
python 2_train_model.py \
    --root_dir /path/to/data \
    --label my_experiment \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --verbose

# Different PCA components (not paper-style, but for experimentation)
python 2_train_model.py \
    --root_dir /path/to/data \
    --label pca_50 \
    --pca_components 50 \
    --overwrite

# Custom train/val/test splits
python 2_train_model.py \
    --root_dir /path/to/data \
    --label custom_split \
    --validation_split 0.15 \
    --test_split 0.15
```

## 📊 Output Interpretation

### Dataset (.npz file)

After Step 1, you'll have a `.npz` file containing:
- `X`: Feature matrix (n_windows × n_channels), RMS per channel
- `y`: Labels for each window (gesture names)
- `class_names`: Sorted list of unique gesture classes
- `emg_fs`: Sampling rate (4000 Hz expected)
- `window_ms`: Window size (250 ms)
- `step_ms`: Step size (250 ms)
- Metadata about channels and preprocessing

### Model Artifacts

After Step 2, you'll have:

**Model weights** (`*_model.pth`):
- PyTorch state dict for the trained neural network

**Preprocessing artifacts**:
- `*_scaler.pkl`: StandardScaler for z-score normalization
- `*_pca.pkl`: PCA transformer (M → 30 dimensions)
- `*_label_encoder.pkl`: Maps gesture names ↔ integer IDs

**Metadata** (`*_metadata.json`):
```json
{
  "model": {
    "architecture": "paper-style",
    "input_dim": 30,
    "output_dim": 10,
    "hidden_layers": [512, 512]
  },
  "training": {
    "epochs": 200,
    "batch_size": 32,
    "best_epoch": 147
  },
  "preprocessing": {
    "pca_components": 30,
    "pca_variance_explained": 0.9234
  },
  "performance": {
    "test_accuracy": 0.9876,
    "test_loss": 0.0543
  }
}
```

**Metrics** (`*_metrics.json`):
- Per-class precision, recall, F1-score
- Confusion matrix
- Training/validation loss curves

## 🔍 Key Differences from Standard Implementation

| Aspect | Paper Approach | Standard Implementation |
|--------|----------------|-------------------------|
| Filter | 120 Hz HP only | 20-498 Hz BP + 60 Hz notch |
| Features | RMS only | 7 features (RMS + MAV + WL + ...) |
| Window | 250 ms, non-overlapping | 200 ms, 50 ms step (overlap) |
| PCA | Always (K=30) | Optional (95% variance) |
| Architecture | 512, 512 hidden | 256, 128 hidden |
| Batch norm | None | Yes |

## 🧪 Example Workflow

Here's a complete example from start to finish:

```bash
# 1. Build dataset from your recordings
python 1_build_training_dataset.py \
    --root_dir ~/emg_data/experiment_2024 \
    --label paper_replication \
    --overwrite \
    --verbose

# Expected output:
# ============================================================
# PAPER-STYLE DATASET BUILDER
# ============================================================
# Found 5 CSV file(s) to process
# [1/5] Processing: recording1.csv
#   Event file: recording1_emg.event
#   Using 64 channels
#   Applying 120 Hz high-pass filter (4th-order Butterworth)
#   Extracting RMS features (window=250ms, step=250ms)
#   Extracted 1234 labeled windows, 64 features
# ...
# SUCCESS! Dataset saved to: ~/emg_data/experiment_2024/paper_replication_training_dataset.npz

# 2. Train the model
python 2_train_model.py \
    --root_dir ~/emg_data/experiment_2024 \
    --label paper_replication \
    --epochs 200 \
    --overwrite \
    --verbose

# Expected output:
# ======================================================================
# PAPER-STYLE MODEL TRAINER
# ======================================================================
# [1/4] Applying z-score normalization...
# [2/4] Applying PCA (M=64 → K=30)...
#   PCA variance explained: 0.9234
# [3/4] Encoding labels...
#   Classes: 10
# [4/4] Splitting data (train/val/test)...
#   Train: 7000 samples
#   Val:   1500 samples
#   Test:  1500 samples
# 
# STARTING TRAINING
# ======================================================================
# Epoch   1/200: train_loss=2.1234, val_loss=1.8765, val_acc=0.3456
# Epoch  10/200: train_loss=0.8765, val_loss=0.7654, val_acc=0.7234
# ...
# Epoch 200/200: train_loss=0.0432, val_loss=0.0512, val_acc=0.9876
# 
# Training complete! Best model at epoch 147
# 
# FINAL RESULTS
# ======================================================================
# Test accuracy: 0.9876
# Test loss: 0.0543
# 
# Classification Report:
#               precision    recall  f1-score   support
# 
#         Rest     0.9900    0.9900    0.9900       150
#         Fist     0.9867    0.9933    0.9900       150
#    ...
# 
# SUCCESS! Training complete.
```

## 🎯 Expected Performance

Based on the paper's results:
- **Able-bodied participants**: ~99.6% accuracy (with 120 Hz filter)
- **SCI participants**: High accuracy maintained (gesture classification)

Your results should be comparable if:
- You have similar data quality (good electrode contact, low impedance)
- Sufficient training data per gesture (paper used multiple trials)
- Proper label alignment (events correctly mark gesture boundaries)

## 🐛 Troubleshooting

### Issue: "No event file found"
**Solution**: Ensure event files are in `<root_dir>/events/` and named like `<recording_stem>_emg.event` or `<recording_stem>.event`

### Issue: "Sample rate differs from expected 4000 Hz"
**Solution**: This is just a warning. The code will still work, but for exact replication, ensure your recordings are at 4000 Hz.

### Issue: "Feature dimension mismatch"
**Solution**: Make sure you're using the same channels for training and inference. Use `--channels` to specify a consistent subset.

### Issue: Low accuracy after training
**Possible causes**:
- Insufficient training data (try collecting more trials)
- Poor electrode contact (check impedances)
- Incorrect label alignment (verify event files match recordings)
- Class imbalance (ensure similar number of samples per gesture)

## 📚 References

If you use this implementation, please cite the original paper:

```bibtex
@article{gesture_classification_paper,
  title={[Insert paper title]},
  author={[Insert authors]},
  journal={Journal of Neural Engineering},
  year={[Insert year]},
  doi={[Insert DOI]}
}
```

## 🔗 Related Files

- `signal_processing_comparison.md` - Detailed comparison of paper vs standard approaches
- Original scripts:
  - `examples/gesture_classifier/1e_build_training_dataset_any.py` - Standard dataset builder
  - `examples/gesture_classifier/2_train_model.py` - Standard model trainer
  - `examples/gesture_classifier/3_predict.py stream` - Maintained real-time inference workflow

## 💡 Tips

1. **Start small**: Test with a single recording first
2. **Check preprocessing**: Visualize filtered signals to ensure quality
3. **Monitor training**: Watch for overfitting (val loss increasing while train loss decreases)
4. **Cross-validation**: For publication-quality results, use k-fold CV (see paper)
5. **Hyperparameter tuning**: The paper's params are good defaults, but may need adjustment for your data

## ❓ Questions?

If you have questions or issues:
1. Check the detailed comparison document: `signal_processing_comparison.md`
2. Review the paper's methods section
3. Compare with the standard implementation in `examples/gesture_classifier/`
