#!/usr/bin/env python3
"""
5_train_model.py - Finger Kinematics Regression Model Training

STEP 5: Train an EMG-to-joint-angle regression model from synchronized dataset.

Prerequisite: Run 4_build_dataset.py to create training dataset with sync offsets applied.

This script:
1. Loads the dataset built by 3_build_dataset.py
2. Configures ModelManager for regression (MSE loss, MAE/R² metrics)
3. Trains EMGRegressor with early stopping
4. Saves model, scaler, PCA (if used), and metrics

Features:
    - Regression mode (continuous joint angle prediction)
    - PCA dimensionality reduction support
    - Early stopping to prevent overfitting
    - Comprehensive metrics: MSE, MAE, R²
    - GPU acceleration (if available)

Examples:
    # Auto-discover dataset
    python 2_train_model.py --root_dir /data
    
    # With PCA reduction
    python 2_train_model.py --root_dir /data --use_pca --pca_variance 0.95
    
    # Custom hyperparameters
    python 2_train_model.py --root_dir /data --epochs 200 --batch_size 32 --learning_rate 0.0005
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from intan.io import (
    load_config_file,
    load_simple_config,
    save_simple_config,
    prompt_directory,
    get_or_prompt_value,
)
from intan.ml import ModelManager, EMGRegressor


def find_dataset_path(root_dir: str, label: str = "") -> str:
    """
    Try common dataset filenames.
    """
    candidates = []
    if label:
        candidates.append(os.path.join(root_dir, "dataset", f"{label}_kinematics_dataset.npz"))
    
    candidates.extend([
        os.path.join(root_dir, "dataset", "kinematics_dataset.npz"),
        os.path.join(root_dir, "kinematics_dataset.npz"),
    ])
    
    for p in candidates:
        if os.path.exists(p):
            return p
    
    raise FileNotFoundError(
        f"No dataset found. Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
    )


def train_model(
    root_dir: str,
    train_npz: str = None,
    label: str = "",
    model_arch: str = "regressor",
    use_pca: bool = False,
    pca_variance: float = 0.95,
    pca_components: int = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10,
    test_size: float = 0.2,
    use_cuda: bool = True,
    seed: int = 42,
    verbose: bool = False,
):
    """
    Train EMG regression model.
    
    Args:
        root_dir: Project root directory
        train_npz: Explicit path to training dataset
        label: Label prefix for model files
        model_arch: Model architecture ('regressor')
        use_pca: Apply PCA dimensionality reduction
        pca_variance: Target explained variance (if pca_components not set)
        pca_components: Explicit number of PCA components
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        patience: Early stopping patience
        test_size: Validation split fraction
        use_cuda: Enable GPU acceleration
        seed: Random seed
        verbose: Verbose logging
    """
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    # Find dataset
    if not train_npz:
        train_npz = find_dataset_path(root_dir, label)
    
    logging.info(f"[INFO] Training regression model")
    logging.info(f"   Root: {root_dir}")
    logging.info(f"   Dataset: {os.path.basename(train_npz)}")
    
    # Load dataset
    data = np.load(train_npz, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Extract metadata
    n_joints = y.shape[1]
    window_ms = float(data.get('window_ms', 200))
    step_ms = float(data.get('step_ms', 50))
    fs = float(data.get('fs', 4000))
    selected_channels = data.get('selected_channels', list(range(128))).tolist()
    channel_names = data.get('channel_names', [f'ch{i}' for i in selected_channels])
    if hasattr(channel_names, 'tolist'):
        channel_names = channel_names.tolist()
    
    logging.info(f"   Loaded: {X.shape[0]} samples, {X.shape[1]} features → {n_joints} joints")
    
    # Joint angle statistics
    logging.info(f"   Joint angle ranges:")
    for joint_idx in range(n_joints):
        min_val, max_val = y[:, joint_idx].min(), y[:, joint_idx].max()
        mean_val, std_val = y[:, joint_idx].mean(), y[:, joint_idx].std()
        logging.info(f"      Joint {joint_idx}: [{min_val:.2f}, {max_val:.2f}], mean={mean_val:.2f}, std={std_val:.2f}")
    
    # Model configuration
    config = {
        "task": "regression",  # KEY: This switches to MSE loss
        "use_pca": use_pca,
        "pca_variance": pca_variance,
        "pca_components": pca_components,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "test_size": test_size,
        "use_cuda": use_cuda,
        "seed": seed,
    }
    
    logging.info(f"   Config:")
    logging.info(f"      Task: regression (MSE loss)")
    logging.info(f"      PCA: {use_pca}")
    if use_pca:
        if pca_components:
            logging.info(f"      PCA components: {pca_components}")
        else:
            logging.info(f"      PCA variance: {pca_variance}")
    logging.info(f"      Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    logging.info(f"      Early stopping patience: {patience}")
    logging.info(f"      Validation split: {test_size}")
    
    # Filter out NaN values
    nan_mask = np.isnan(y).any(axis=1)
    if nan_mask.sum() > 0:
        logging.info(f"\n[Filtering NaN values...]")
        logging.info(f"   Removing {nan_mask.sum()} samples with NaN ({100*nan_mask.sum()/len(y):.2f}%)")
        X = X[~nan_mask]
        y = y[~nan_mask]
        logging.info(f"   Remaining: {len(X)} samples")
    
    # Filter out outliers (joint angles outside physiological range 0-180 degrees)
    outlier_mask = ((y < 0) | (y > 180)).any(axis=1)
    if outlier_mask.sum() > 0:
        logging.info(f"\n[Filtering outliers...]")
        logging.info(f"   Removing {outlier_mask.sum()} samples with angles outside [0, 180]° ({100*outlier_mask.sum()/len(y):.2f}%)")
        X = X[~outlier_mask]
        y = y[~outlier_mask]
        logging.info(f"   Remaining: {len(X)} samples")
    
    # Select model class
    if model_arch == "regressor":
        model_cls = EMGRegressor
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    
    # Initialize ModelManager
    manager = ModelManager(
        root_dir=root_dir,
        label=label,
        model_cls=model_cls,
        config=config,
        verbose=verbose
    )
    
    # Train
    logging.info(f"\n[Training...]")
    manager.train(X, y)
    
    # Save metadata for prediction
    metadata = {
        'n_joints': n_joints,
        'input_dim': X.shape[1],
        'output_dim': n_joints,
        'selected_channels': selected_channels,
        'channel_names': channel_names,
        'window_ms': window_ms,
        'step_ms': step_ms,
        'fs': fs,
        'task': 'regression',
        'model_arch': model_arch,
    }
    manager.save_metadata(metadata)
    logging.info(f"[INFO] Metadata saved")
    
    # Get validation predictions
    y_val_pred = manager.y_val_pred
    y_val_true = manager.y_val_true
    
    # Compute per-joint metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    logging.info(f"\n[Validation Metrics]")
    logging.info(f"   Overall:")
    overall_mse = mean_squared_error(y_val_true, y_val_pred)
    overall_mae = mean_absolute_error(y_val_true, y_val_pred)
    overall_r2 = r2_score(y_val_true, y_val_pred)
    logging.info(f"      MSE:  {overall_mse:.4f}")
    logging.info(f"      MAE:  {overall_mae:.4f}")
    logging.info(f"      R²:   {overall_r2:.4f}")
    
    logging.info(f"\n   Per-Joint:")
    for joint_idx in range(n_joints):
        mse = mean_squared_error(y_val_true[:, joint_idx], y_val_pred[:, joint_idx])
        mae = mean_absolute_error(y_val_true[:, joint_idx], y_val_pred[:, joint_idx])
        r2 = r2_score(y_val_true[:, joint_idx], y_val_pred[:, joint_idx])
        logging.info(f"      Joint {joint_idx}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    logging.info(f"\n[OK] Model saved to: {manager.model_dir}")
    logging.info(f"   Files:")
    logging.info(f"      - model.pth (weights)")
    logging.info(f"      - scaler.pkl (normalization)")
    if use_pca:
        logging.info(f"      - pca.pkl (dimensionality reduction)")
    logging.info(f"      - metadata.json (configuration)")
    logging.info(f"      - metrics.json (performance)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--root_dir", help="Project root directory")
    parser.add_argument("--train_npz", help="Path to training dataset NPZ")
    parser.add_argument("--label", default="", help="Label prefix for outputs")
    parser.add_argument("--model_arch", default="regressor", choices=["regressor"])
    parser.add_argument("--use_pca", action="store_true", help="Enable PCA")
    parser.add_argument("--pca_variance", type=float, default=0.95)
    parser.add_argument("--pca_components", type=int, help="Explicit PCA components")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_file", help="Load config from file")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_simple_config(Path(__file__).parent / ".kinematics_config")
    if args.config_file:
        cfg.update(load_config_file(args.config_file))
    
    # Prompt for root_dir if not provided
    if not args.root_dir:
        root_dir, was_prompted = get_or_prompt_value(
            arg_value=None,
            config=cfg,
            key='root_dir',
            prompt_func=prompt_directory,
            title="Select Project Root Directory",
            initial_dir=cfg.get('root_dir') if 'root_dir' in cfg else None
        )
        
        if not root_dir:
            print("Operation cancelled by user")
            sys.exit(0)
        
        args.root_dir = root_dir
        if was_prompted:
            cfg['root_dir'] = root_dir
            save_simple_config(cfg, Path(__file__).parent / ".kinematics_config")
    
    # Merge CLI args with config
    params = {
        k: getattr(args, k) if getattr(args, k) is not None else cfg.get(k, v)
        for k, v in {
            "root_dir": None,
            "train_npz": None,
            "label": "",
            "model_arch": "regressor",
            "use_pca": False,
            "pca_variance": 0.95,
            "pca_components": None,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "patience": 10,
            "test_size": 0.2,
            "use_cuda": True,
            "seed": 42,
            "verbose": False,
        }.items()
    }
    
    # Save updated config
    for k in ["use_pca", "pca_variance", "pca_components", "epochs", "batch_size", "learning_rate", "patience"]:
        if k in params:
            cfg[k] = params[k]
    save_simple_config(cfg, Path(__file__).parent / ".kinematics_config")
    
    try:
        train_model(**params)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
