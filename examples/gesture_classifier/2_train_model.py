#!/usr/bin/env python3
"""
2_train_model.py

Train an EMG gesture classifier from a pre-built dataset (.npz).

Modes:
    1. Auto-discover: Finds dataset in --root_dir (default)
    2. Explicit files: Use --train_npz for one or more training datasets
    3. External test: Add --test_npz for held-out evaluation
    
Features:
    - Multi-dataset concatenation
    - External test set support
    - Save tagging for experiment tracking
    - K-fold cross-validation
    - Comprehensive metrics export

Examples:
    # Simple: auto-discover dataset
    python 2_train_model.py --root_dir /data --label gesture
    
    # Multiple datasets concatenated
    python 2_train_model.py --root_dir /data --train_npz dataset1.npz dataset2.npz
    
    # External test set
    python 2_train_model.py --root_dir /data --train_npz train.npz --test_npz test.npz
    
    # Experiment tracking with tags
    python 2_train_model.py --root_dir /data --train_npz train.npz --save_tag "exp_v2"
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from intan.io import load_config_file, load_simple_config, save_simple_config, prompt_directory, prompt_file, get_or_prompt_value
from sklearn.metrics import classification_report, confusion_matrix


def _resolve_paths(paths, root_dir):
    """Resolve relative paths relative to root_dir."""
    out = []
    for p in (paths or []):
        if os.path.isabs(p):
            out.append(p)
        else:
            out.append(os.path.join(root_dir, p))
    return out


def load_npz_list(npz_paths):
    """
    Load and concatenate multiple NPZ datasets.
    
    Args:
        npz_paths: List of paths to NPZ files
        
    Returns:
        tuple: (X, y, metas) where X and y are concatenated, metas is list of metadata dicts
    """
    Xs, ys, metas = [], [], []
    for p in npz_paths:
        logging.info(f"   Loading: {os.path.basename(p)}")
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["y"])
        metas.append({
            "path": p,
            "class_names": d["class_names"].tolist() if "class_names" in d.files else None,
            "label_to_id": json.loads(str(d["label_to_id_json"].item())) if "label_to_id_json" in d.files else None,
            "window_ms": int(d["window_ms"]) if "window_ms" in d.files else None,
            "step_ms": int(d["step_ms"]) if "step_ms" in d.files else None,
            "modality": str(d["modality"].item()) if "modality" in d.files else None,
            "emg_fs": float(d["emg_fs"]) if "emg_fs" in d.files else None,
        })
    
    # Sanity check: feature dimensions must match
    fdim = {x.shape[1] for x in Xs}
    if len(fdim) != 1:
        raise ValueError(f"Feature dimensions differ across datasets: {fdim}")
    
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    
    logging.info(f"   Combined: {X.shape[0]} samples from {len(npz_paths)} dataset(s)")
    return X, y, metas


def _find_dataset_path(root_dir: str, label: str | None) -> str:
    """
    Try common dataset filenames. Prefer label-specific, then generic, then emg/ subdir.
    """
    candidates = []
    if label:
        candidates.append(os.path.join(root_dir, f"{label}_training_dataset.npz"))
    candidates += [
        os.path.join(root_dir, "training_dataset.npz"),
        os.path.join(root_dir, "dataset_emg_windows.npz"),
        os.path.join(root_dir, "emg", "dataset_emg_windows.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # default to first candidate (label-prefixed) for the error message
    return candidates[0] if candidates else os.path.join(root_dir, "training_dataset.npz")


def _load_dataset(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as d:
        X = d["X"]
        # Prefer integer y_id if available; else use y (strings/objects)
        if "y_id" in d.files:
            y = d["y_id"]
        else:
            y = d["y"]

        # Optional metadata
        emg_fs   = float(d["emg_fs"])   if "emg_fs"   in d.files else None
        window_ms= int(d["window_ms"])  if "window_ms" in d.files else None
        step_ms  = int(d["step_ms"])    if "step_ms"   in d.files else None

        # feature_spec saved as a JSON string; handle scalar/object ndarray
        feature_spec = None
        if "feature_spec" in d.files:
            raw = d["feature_spec"]
            try:
                raw = raw.item() if getattr(raw, "shape", ()) == () else raw
            except Exception:
                pass
            feature_spec = json.loads(str(raw))

        selected_channels = d["selected_channels"].tolist() if "selected_channels" in d.files else []
        channel_names     = d["channel_names"].tolist()     if "channel_names"     in d.files else []
        class_names       = d["class_names"].tolist()       if "class_names"       in d.files else None

        label_to_id = None
        if "label_to_id_json" in d.files:
            try:
                label_to_id = json.loads(str(d["label_to_id_json"].item()))
            except Exception:
                label_to_id = json.loads(str(d["label_to_id_json"]))

    return {
        "X": X, "y": y,
        "emg_fs": emg_fs, "window_ms": window_ms, "step_ms": step_ms,
        "feature_spec": feature_spec,
        "selected_channels": selected_channels,
        "channel_names": channel_names,
        "class_names": class_names,
        "label_to_id": label_to_id,
    }


def _infer_label_classes(y, class_names, label_to_id):
    """
    Derive ordered label class names for metadata.
    Priority: class_names in dataset > label_to_id order > unique(y)
    """
    if class_names:
        return [str(c) for c in class_names]
    if label_to_id:
        inv = {v: k for k, v in label_to_id.items()}
        return [str(inv[i]) for i in range(len(inv))]
    # fallback
    return [str(c) for c in np.unique(y)]


def train_model(cfg: dict, save_eval: bool = False):
    try:
        from intan.ml import ModelManager, EMGClassifier, EMGClassifierCNNLSTM
    except ImportError as exc:
        raise RuntimeError(
            "Model training requires the ML extra: pip install 'python-intan[ml]'"
        ) from exc
    """
    Train an EMG gesture classifier.
    
    Args:
        cfg: Configuration dictionary with keys:
            - root_dir: Root directory for model/data
            - label: Label prefix for saved files
            - train_npz: Optional list of NPZ files to concatenate
            - test_npz: Optional external test set
            - save_tag: Optional suffix for saved artifacts
            - kfold: Use k-fold cross-validation
            - overwrite: Retrain even if model exists
        save_eval: Save validation predictions/metrics to JSON
    """
    label = cfg.get("label", "")
    root_dir = cfg["root_dir"]
    save_tag = cfg.get("save_tag", "").strip()
    
    # Determine data source: explicit files or auto-discover
    use_external_test = False
    X_test = y_test = None
    
    if cfg.get("train_npz"):
        # Mode 1: Explicit training file(s)
        train_list = _resolve_paths(cfg["train_npz"], root_dir)
        logging.info(f"Loading training data from {len(train_list)} file(s):")
        X, y, metas = load_npz_list(train_list)
        
        # Extract metadata from first file
        first = metas[0] if metas else {}
        meta_dict = {
            "emg_fs": first.get("emg_fs"),
            "window_ms": first.get("window_ms"),
            "step_ms": first.get("step_ms"),
            "feature_spec": None,
            "selected_channels": [],
            "channel_names": [],
            "class_names": first.get("class_names"),
        }
        
        # Optional: load external test set
        if cfg.get("test_npz"):
            test_path = _resolve_paths([cfg["test_npz"]], root_dir)[0]
            logging.info(f"Loading external test set: {os.path.basename(test_path)}")
            with np.load(test_path, allow_pickle=True) as d:
                X_test = d["X"]
                y_test = d["y"]
                if X_test.shape[1] != X.shape[1]:
                    raise ValueError(f"Train/test feature dimension mismatch: {X.shape[1]} vs {X_test.shape[1]}")
            use_external_test = True
            logging.info(f"   External test: {X_test.shape[0]} samples")
    else:
        # Mode 2: Auto-discover single dataset in root_dir
        data_path = _find_dataset_path(root_dir, label)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found. Looked for: {data_path}")
        
        logging.info(f"Loading dataset: {os.path.basename(data_path)}")
        meta_dict = _load_dataset(data_path)
        X, y = meta_dict.pop("X"), meta_dict.pop("y")
    
    logging.info(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    logging.info(f"   Classes: {sorted(set(y))}")
    
    # ========== MODEL SELECTION ==========
    # Choose model architecture via --model_type argument:
    #   'cnn':      Simple fully-connected CNN (default)
    #   'cnn_lstm': Hybrid CNN-LSTM for spatiotemporal learning
    # =====================================
    model_type = cfg.get("model_type", "cnn").lower()
    if model_type == "cnn_lstm":
        logging.info("Using CNN-LSTM hybrid model")
        model_cls = EMGClassifierCNNLSTM
    else:
        logging.info("Using standard CNN classifier")
        model_cls = EMGClassifier
    
    # Configure PCA for dimensionality reduction
    cfg['use_pca'] = True
    cfg['pca_components'] = 30  # Keep top 30 principal components
    
    # Configure training hyperparameters for better progress feedback
    if 'hyperparameters' not in cfg:
        cfg['hyperparameters'] = {}
    
    # Adjust for CNN-LSTM (more complex, needs more careful training)
    if model_type == "cnn_lstm":
        cfg['hyperparameters'].setdefault('num_epochs', 1000)  # Reduced from 3000
        cfg['hyperparameters'].setdefault('val_interval', 10)  # Show progress every 10 epochs
        cfg['hyperparameters'].setdefault('learning_rate', 5e-4)  # Slightly lower LR
    else:
        cfg['hyperparameters'].setdefault('num_epochs', 1000)
        cfg['hyperparameters'].setdefault('val_interval', 10)
    
    # Initialize model manager
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=model_cls, config=cfg)

    # Training mode selection
    if cfg.get("kfold", False):
        # K-fold cross-validation
        logging.info("Running k-fold cross-validation...")
        logging.info("This may take several minutes...")
        sys.stdout.flush()
        cv_metrics = manager.cross_validate(X, y)
        
        # Extract accuracy from each fold
        per_fold = []
        for i, m in enumerate(cv_metrics, 1):
            acc = None
            if isinstance(m, dict):
                acc = m.get("accuracy")
                if acc is None:
                    cr = m.get("classification_report")
                    if isinstance(cr, dict):
                        acc = cr.get("accuracy")
            per_fold.append(float(acc) if acc is not None else float("nan"))
            logging.info(f"   Fold {i}: accuracy = {acc:.4f}" if acc else f"   Fold {i}: N/A")
        
        vals = np.array(per_fold, dtype=float)
        mean = float(np.nanmean(vals)) if vals.size else float("nan")
        std = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        logging.info(f"Mean accuracy: {mean:.4f} +/- {std:.4f}")
        
        # Save CV summary
        out_dir = os.path.join(root_dir, "model")
        os.makedirs(out_dir, exist_ok=True)
        tag = f"_{save_tag}" if save_tag else ""
        out_path = os.path.join(out_dir, f'{label}_cv_summary{tag}.json'.strip("_"))
        
        payload = {
            "metric_name": "accuracy",
            "per_fold": vals.tolist(),
            "mean": mean,
            "std": std,
            "all_metrics": cv_metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info(f"Saved CV summary: {out_path}")

    elif not os.path.isfile(manager.model_path) or cfg.get("overwrite", False):
        if use_external_test:
            logging.info("Training with external test set...")
            epochs = cfg.get('hyperparameters', {}).get('num_epochs', 1000)
            val_int = cfg.get('hyperparameters', {}).get('val_interval', 10)
            logging.info(f"Starting training: {epochs} epochs, progress every {val_int} epochs")
            sys.stdout.flush()
            manager.train(X, y)
            
            # Evaluate on external test set
            logging.info("Evaluating on external test set...")
            y_pred = manager.predict(X_test)
            labels_sorted = np.unique(np.concatenate([y_test, y_pred]))
            
            report = classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
            
            test_acc = report.get("accuracy", 0.0)
            logging.info(f"Test accuracy: {test_acc:.4f}")
            
            # Save external test results
            out_dir = os.path.join(root_dir, "model")
            os.makedirs(out_dir, exist_ok=True)
            tag = f"_{save_tag}" if save_tag else ""
            out_path = os.path.join(out_dir, f"{label}_external_test{tag}.json".strip("_"))
            
            out = {
                "labels": labels_sorted.tolist(),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "y_true": np.asarray(y_test, dtype=object).tolist(),
                "y_pred": np.asarray(y_pred, dtype=object).tolist(),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            logging.info(f"Saved external test results: {out_path}")
        else:
            logging.info("Training with train/val split...")
            epochs = cfg.get('hyperparameters', {}).get('num_epochs', 1000)
            val_int = cfg.get('hyperparameters', {}).get('val_interval', 10)
            logging.info(f"Starting training: {epochs} epochs, progress every {val_int} epochs")
            sys.stdout.flush()  # Force output to display
            manager.train(X, y)
    else:
        logging.info("Model exists, loading...")
        manager.load_model()

    # Label classes (don’t assume LabelEncoder exists)
    if hasattr(manager, "label_encoder"):
        label_classes = [str(c) for c in manager.label_encoder.classes_]
    elif hasattr(manager, "classes_"):
        label_classes = [str(c) for c in manager.classes_]
    else:
        label_classes = sorted([str(c) for c in np.unique(y)])


    # Scaler snapshot (if your pipeline includes a StandardScaler inside ModelManager)
    scaler_mean = getattr(getattr(manager, "scaler", None), "mean_", None)
    scaler_scale = getattr(getattr(manager, "scaler", None), "scale_", None)
    scaler_mean = None if scaler_mean is None else scaler_mean.tolist()
    scaler_scale = None if scaler_scale is None else scaler_scale.tolist()

    meta = manager.build_metadata(
        sample_rate_hz=meta_dict.get("emg_fs"),
        window_ms=meta_dict.get("window_ms"),
        step_ms=meta_dict.get("step_ms"),
        envelope_cutoff_hz=cfg.get("envelope_cutoff_hz", 5.0),
        selected_channels=meta_dict.get("selected_channels"),
        channel_names=meta_dict.get("channel_names"),
        feature_spec=meta_dict.get("feature_spec"),
        n_features=X.shape[1],
        label_classes=label_classes,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        extra=getattr(manager, "eval_metrics", None),
    )
    manager.save_metadata(meta)
    logging.info("Training complete. Metadata saved.")

    # Print evaluation metrics if available
    if hasattr(manager, "eval_metrics") and manager.eval_metrics:
        logging.info(f"   Final metrics: {manager.eval_metrics}")

    if save_eval:
        # Try to pull validation arrays from the manager
        y_true_val = getattr(manager, "y_val_true", None)
        y_pred_val = getattr(manager, "y_val_pred", None)
        if y_true_val is not None and y_pred_val is not None:
            labels_sorted = np.unique(np.concatenate([y_true_val, y_pred_val]))
            rep_dict = classification_report(y_true_val, y_pred_val, labels=labels_sorted,
                                             zero_division=0, output_dict=True)
            cm = confusion_matrix(y_true_val, y_pred_val, labels=labels_sorted)
            out = {
                "labels": labels_sorted.tolist(),
                "confusion_matrix": cm.tolist(),
                "classification_report": rep_dict,
                "y_true_val": np.asarray(y_true_val, dtype=object).tolist(),
                "y_pred_val": np.asarray(y_pred_val, dtype=object).tolist(),
                "val_indices": getattr(manager, "val_indices", None).tolist()
                if getattr(manager, "val_indices", None) is not None else None,
            }
            out_dir = os.path.join(root_dir, "model")
            os.makedirs(out_dir, exist_ok=True)
            tag = f"_{save_tag}" if save_tag else ""
            out_path = os.path.join(out_dir, f"{label}_val_eval{tag}.json".strip("_"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            logging.info(f"Saved validation evaluation: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Train an EMG gesture classification model.")
    p.add_argument("--config_file", type=str, default=None, help="Path to config file.")
    p.add_argument("--root_dir", type=str, default=None, help="Root directory (prompts if not provided).")
    p.add_argument("--dataset_dir", type=str, default=None, help="Directory containing dataset files (deprecated, use root_dir).")
    p.add_argument("--train_npz", nargs="+", default=None, help="One or more training .npz files (instead of auto-discovery).")
    p.add_argument("--test_npz", type=str, default=None, help="Optional external test .npz. If set, skip internal split.")
    p.add_argument("--save_tag", type=str, default="", help="Extra tag for model/metrics filenames.")
    p.add_argument("--label", type=str, default="", help="Label prefix for model/dataset files.")
    p.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "cnn_lstm"], help="Model architecture: 'cnn' (default) or 'cnn_lstm'.")
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 1000).")
    p.add_argument("--val_interval", type=int, default=None, help="Validation check interval in epochs (default: 10).")
    p.add_argument("--kfold", action="store_true", help="Use k-fold cross-validation instead of train/test split.")
    p.add_argument("--overwrite", action="store_true", help="Retrain model even if one exists.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    p.add_argument("--save_eval", action="store_true", help="After training, save the validation-set predictions/metrics to JSON for plotting.")
    args = p.parse_args()

    # Load shared config first, then override with specific config_file if provided
    cfg = load_simple_config(Path(__file__).parent / ".gesture_config")
    if args.config_file:
        cfg.update(load_config_file(args.config_file))
    
    # Prompt for root_dir if not provided
    root_dir = args.root_dir
    if not root_dir:
        root_dir, was_prompted = get_or_prompt_value(
            arg_value=None,
            config=cfg,
            key='root_dir',
            prompt_func=prompt_directory,
            title="Select Project Root Directory",
            initial_dir=cfg.get('root_dir') if 'root_dir' in cfg else None
        )
        # Save to config immediately
        if was_prompted:
            cfg['root_dir'] = root_dir
            save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Training Configuration")
            print("[*] Saved root_dir to .gesture_config")
    else:
        # Save CLI-provided root_dir to config if not already there
        if 'root_dir' not in cfg or cfg.get('root_dir') != root_dir:
            cfg['root_dir'] = root_dir
            save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Training Configuration")
            print("[*] Saved root_dir to .gesture_config")
    
    # Update config with command line arguments
    cfg.update({
        "root_dir": root_dir or cfg.get("root_dir", ""),
        "dataset_dir": args.dataset_dir or args.root_dir or cfg.get("dataset_dir", ""),
        "train_npz": args.train_npz or cfg.get("train_npz", None),
        "test_npz": args.test_npz or cfg.get("test_npz", None),
        "save_tag": args.save_tag or cfg.get("save_tag", ""),
        "label": args.label or cfg.get("label", ""),
        "model_type": args.model_type or cfg.get("model_type", "cnn"),
        "kfold": args.kfold or cfg.get("kfold", False),
        "overwrite": args.overwrite or cfg.get("overwrite", False),
        "verbose": args.verbose or cfg.get("verbose", False),
    })
    
    # Override hyperparameters if specified
    if args.epochs is not None or args.val_interval is not None:
        if 'hyperparameters' not in cfg:
            cfg['hyperparameters'] = {}
        if args.epochs is not None:
            cfg['hyperparameters']['num_epochs'] = args.epochs
        if args.val_interval is not None:
            cfg['hyperparameters']['val_interval'] = args.val_interval

    train_model(cfg, args.save_eval or cfg.get("save_eval", False))

if __name__ == "__main__":
    main()
