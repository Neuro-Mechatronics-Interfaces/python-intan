#!/usr/bin/env python3
"""
2_train_model.py

Train an EMG gesture classifier following the published paper's approach:
- PCA dimensionality reduction to K=30 components
- Neural network: 30 → 512 → 512 → 10
- No batch normalization (paper doesn't mention it)
- Dropout 0.2 after each hidden layer
- Adam optimizer
- 200 epochs, batch size 32
- Categorical cross-entropy loss

This script replicates the exact model architecture and training procedure from the
Journal of Neural Engineering paper.

Examples
--------
python 2_train_model.py \
    --root_dir /path/to/data \
    --label paper_replication \
    --overwrite --verbose

python 2_train_model.py \
    --root_dir /path/to/data \
    --label paper_style \
    --epochs 200 \
    --pca_components 30
"""

import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


class PaperStyleEMGClassifier(nn.Module):
    """
    Neural network architecture matching the published paper:

    Input: 30 dimensions (after PCA)
    Hidden layer 1: 512 nodes + Dropout(0.2)
    Hidden layer 2: 512 nodes + Dropout(0.2)
    Output: N classes (softmax via CrossEntropyLoss)

    Key differences from standard implementation:
    - NO batch normalization (paper doesn't mention it)
    - Fixed 512-node hidden layers (not 256/128)
    - Designed for PCA-reduced input (K=30)
    """

    def __init__(self, input_dim=30, output_dim=10):
        super(PaperStyleEMGClassifier, self).__init__()

        logging.info(f"Building paper-style model: {input_dim}→512→512→{output_dim}")

        self.model = nn.Sequential(
            # First hidden layer: 512 nodes
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Second hidden layer: 512 nodes
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Output layer (no activation - handled by CrossEntropyLoss)
            nn.Linear(512, output_dim)
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


def train_paper_style_model(
        root_dir: str,
        label: str = "paper_style",
        dataset_path: str | None = None,
        pca_components: int = 30,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42,
        overwrite: bool = False,
        verbose: bool = False,
):
    """
    Train gesture classifier following the paper's exact approach.

    Training Pipeline (matches paper):
    1. Load preprocessed dataset (RMS features from 120 Hz filtered data)
    2. Z-score normalization (StandardScaler)
    3. PCA dimensionality reduction to K=30
    4. Train/validation/test split (64/16/20 in paper, configurable here)
    5. Neural network training: 30→512→512→N
    6. Adam optimizer, categorical cross-entropy loss
    7. 200 epochs with batch size 32
    8. Model selection based on lowest validation loss

    Parameters
    ----------
    root_dir : str
        Root directory for saving model artifacts
    label : str
        Label prefix for model files
    dataset_path : str, optional
        Path to .npz dataset. If None, looks for <root_dir>/<label>_training_dataset.npz
    pca_components : int
        Number of PCA components (paper uses K=30)
    epochs : int
        Training epochs (paper uses 200)
    batch_size : int
        Batch size (paper uses 32)
    learning_rate : float
        Adam learning rate
    validation_split : float
        Fraction for validation set
    test_split : float
        Fraction for test set
    random_seed : int
        Random seed for reproducibility
    overwrite : bool
        Whether to overwrite existing model
    verbose : bool
        Enable verbose logging
    """

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    # Set random seeds for reproducibility (as in paper)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    logging.info("=" * 70)
    logging.info("PAPER-STYLE MODEL TRAINER")
    logging.info("=" * 70)
    logging.info(f"Replicating training procedure from Journal of Neural Engineering paper:")
    logging.info(f"  - PCA: M channels → {pca_components} components")
    logging.info(f"  - Architecture: {pca_components}→512→512→N (no batch norm)")
    logging.info(f"  - Dropout: 0.2 after each hidden layer")
    logging.info(f"  - Optimizer: Adam (lr={learning_rate})")
    logging.info(f"  - Epochs: {epochs}, Batch size: {batch_size}")
    logging.info(f"  - Loss: Categorical cross-entropy")
    logging.info("=" * 70)

    # Setup paths
    model_dir = os.path.join(root_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    prefix = f"{label}_" if label else ""
    model_path = os.path.join(model_dir, f"{prefix}model.pth")
    scaler_path = os.path.join(model_dir, f"{prefix}scaler.pkl")
    pca_path = os.path.join(model_dir, f"{prefix}pca.pkl")
    encoder_path = os.path.join(model_dir, f"{prefix}label_encoder.pkl")
    metadata_path = os.path.join(model_dir, f"{prefix}metadata.json")
    metrics_path = os.path.join(model_dir, f"{prefix}metrics.json")

    if os.path.exists(model_path) and not overwrite:
        raise FileExistsError(f"Model exists at {model_path}. Use --overwrite to replace.")

    # Load dataset
    if dataset_path is None:
        dataset_path = os.path.join(root_dir, f"{label}_training_dataset.npz")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    logging.info(f"\nLoading dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    emg_fs = float(data["emg_fs"])
    window_ms = int(data["window_ms"])
    step_ms = int(data["step_ms"])
    class_names = data["class_names"].tolist()

    # Load channel information if available
    channel_names = None
    if "channel_names" in data:
        channel_names = data["channel_names"].tolist()
    elif "selected_channels" in data:
        # Fallback: generate from channel indices
        channel_names = [f"CH{i}" for i in data["selected_channels"]]

    logging.info(f"Dataset loaded: X shape={X.shape}, y shape={y.shape}")
    logging.info(f"Classes: {class_names}")
    logging.info(f"Sample rate: {emg_fs} Hz, Window: {window_ms} ms, Step: {step_ms} ms")
    if channel_names:
        logging.info(f"Channels: {len(channel_names)} ({channel_names[:3]}...{channel_names[-3:]})")

    # ========================================================================
    # PAPER-SPECIFIC PREPROCESSING PIPELINE
    # ========================================================================

    # 1. Z-score normalization (as in paper Eq. 4)
    logging.info("\n[1/4] Applying z-score normalization...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info(f"  Mean: {scaler.mean_[:5]}... (first 5)")
    logging.info(f"  Std: {scaler.scale_[:5]}... (first 5)")

    # 2. PCA dimensionality reduction to K=30 (as in paper Eq. 5-6)
    logging.info(f"\n[2/4] Applying PCA (M={X.shape[1]} → K={pca_components})...")
    pca = PCA(n_components=pca_components, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = np.sum(pca.explained_variance_ratio_)
    logging.info(f"  PCA variance explained: {explained_var:.4f}")
    logging.info(f"  New shape: {X_pca.shape}")

    # 3. Label encoding
    logging.info(f"\n[3/4] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    logging.info(f"  Classes: {n_classes}")
    logging.info(f"  Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

    # 4. Train/val/test split
    # Paper mentions 64-16-20 split, we'll do a two-stage split to approximate
    logging.info(f"\n[4/4] Splitting data (train/val/test)...")

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_pca, y_encoded,
        test_size=test_split,
        random_state=random_seed,
        stratify=y_encoded
    )

    # Second split: separate validation from training
    val_fraction = validation_split / (1 - test_split)  # Adjust for already removed test set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction,
        random_state=random_seed,
        stratify=y_train_val
    )

    logging.info(f"  Train: {X_train.shape[0]} samples")
    logging.info(f"  Val:   {X_val.shape[0]} samples")
    logging.info(f"  Test:  {X_test.shape[0]} samples")

    # ========================================================================
    # MODEL TRAINING (Paper's approach)
    # ========================================================================

    logging.info("\n" + "=" * 70)
    logging.info("STARTING TRAINING")
    logging.info("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model (paper's architecture)
    model = PaperStyleEMGClassifier(
        input_dim=pca_components,
        output_dim=n_classes
    ).to(device)

    # Optimizer (Adam, as in paper)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function (categorical cross-entropy, as in paper Eq. 8)
    criterion = nn.CrossEntropyLoss()

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    logging.info(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()

        # Mini-batch training
        n_samples = X_train.shape[0]
        indices = torch.randperm(n_samples)

        epoch_train_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            X_batch = X_train_t[batch_indices]
            y_batch = y_train_t[batch_indices]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_train_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_losses.append(val_loss)

            # Calculate validation accuracy
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        # Save best model (based on validation loss, as in paper)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            logging.info(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

    logging.info(f"\nTraining complete! Best model at epoch {best_epoch + 1}")

    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================

    # Load best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test set evaluation
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t).item()
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()

    logging.info("\n" + "=" * 70)
    logging.info("FINAL RESULTS")
    logging.info("=" * 70)
    logging.info(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Test loss: {test_loss:.4f}")

    # Per-class accuracy (similar to paper's classification report)
    from sklearn.metrics import classification_report, confusion_matrix

    test_preds_np = test_preds.cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()

    report = classification_report(
        y_test_np, test_preds_np,
        target_names=label_encoder.classes_,
        digits=4
    )
    logging.info("\nClassification Report:")
    logging.info("\n" + report)

    cm = confusion_matrix(y_test_np, test_preds_np)
    logging.info("\nConfusion Matrix:")
    logging.info(cm)

    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================

    logging.info("\n" + "=" * 70)
    logging.info("SAVING MODEL ARTIFACTS")
    logging.info("=" * 70)

    # Save scaler
    joblib.dump(scaler, scaler_path)
    logging.info(f"✓ Scaler saved: {scaler_path}")

    # Save PCA
    joblib.dump(pca, pca_path)
    logging.info(f"✓ PCA saved: {pca_path}")

    # Save label encoder
    joblib.dump(label_encoder, encoder_path)
    logging.info(f"✓ Label encoder saved: {encoder_path}")

    # Model already saved during training
    logging.info(f"✓ Model saved: {model_path}")

    # Save metadata
    metadata = {
        "schema_version": "1.0.0-paper-style",
        "model": {
            "architecture": "paper-style",
            "class": "PaperStyleEMGClassifier",
            "input_dim": pca_components,
            "output_dim": n_classes,
            "hidden_layers": [512, 512],
            "dropout": 0.2,
            "batch_normalization": False,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "best_epoch": best_epoch + 1,
        },
        "preprocessing": {
            "normalization": "z-score (StandardScaler)",
            "pca_components": pca_components,
            "pca_variance_explained": float(explained_var),
        },
        "data": {
            "sample_rate_hz": float(emg_fs),
            "window_ms": int(window_ms),
            "step_ms": int(step_ms),
            "feature_type": "RMS-only",
            "filter_type": "120 Hz high-pass (4th-order Butterworth)",
            "selected_channel_names": channel_names if channel_names else [],
            "n_channels": int(X.shape[1]) if channel_names is None else len(channel_names),
        },
        "labels": {
            "classes": label_encoder.classes_.tolist(),
            "n_classes": n_classes,
        },
        "performance": {
            "best_val_loss": float(best_val_loss),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
        },
        "splits": {
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"✓ Metadata saved: {metadata_path}")

    # Save detailed metrics
    metrics = {
        "classification_report": classification_report(
            y_test_np, test_preds_np,
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"✓ Metrics saved: {metrics_path}")

    logging.info("\n" + "=" * 70)
    logging.info("SUCCESS! Training complete.")
    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Model artifacts saved to: {model_dir}")
    logging.info("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train gesture classifier following the published paper's approach"
    )
    p.add_argument("--root_dir", type=str, required=True,
                   help="Root directory for model artifacts")
    p.add_argument("--label", type=str, default="paper_style",
                   help="Label prefix for model files")
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Path to .npz dataset (default: <root_dir>/<label>_training_dataset.npz)")
    p.add_argument("--pca_components", type=int, default=30,
                   help="Number of PCA components (paper uses K=30)")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs (paper uses 200)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (paper uses 32)")
    p.add_argument("--learning_rate", type=float, default=0.001,
                   help="Adam learning rate")
    p.add_argument("--validation_split", type=float, default=0.2,
                   help="Validation set fraction")
    p.add_argument("--test_split", type=float, default=0.1,
                   help="Test set fraction")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing model")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging")

    args = p.parse_args()

    train_paper_style_model(
        root_dir=args.root_dir,
        label=args.label,
        dataset_path=args.dataset_path,
        pca_components=args.pca_components,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        test_split=args.test_split,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
