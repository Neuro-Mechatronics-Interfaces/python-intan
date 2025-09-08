#!/usr/bin/env python3
"""
2_train_model.py

Train an EMG gesture classifier from a pre-built dataset (.npz).
Compatible with datasets produced by 1_build_training_dataset_from_npz.py
"""

import os
import json
import argparse
import numpy as np
from intan.io import load_config_file
from intan.ml import ModelManager, EMGClassifier
from sklearn.metrics import classification_report, confusion_matrix


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
    label = cfg.get("label", "")
    data_path = _find_dataset_path(cfg["root_dir"], label)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found. Looked for: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"unique Y: {np.unique(y)}")

    feature_spec = None
    if "feature_spec" in data.files:
        feature_spec = json.loads(str(data["feature_spec"]))

    manager = ModelManager(root_dir=cfg["root_dir"], label=label, model_cls=EMGClassifier, config=cfg)

    if cfg.get("kfold", False):
        cv_metrics = manager.cross_validate(X, y)
        print("Cross-validation metrics:")
        for i, m in enumerate(cv_metrics, 1):
            print(f"Fold {i}: {m}")
    elif not os.path.isfile(manager.model_path) or cfg.get("overwrite", False):
        print("Training new model...")
        manager.train(X, y)
    else:
        print("Model exists; loading.")
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
        sample_rate_hz=data["emg_fs"],
        window_ms=data["window_ms"],
        step_ms=data["step_ms"],
        envelope_cutoff_hz=cfg.get("envelope_cutoff_hz", 5.0),
        selected_channels=data["selected_channels"],
        channel_names=data["channel_names"],
        feature_spec=data["feature_spec"],
        n_features=X.shape[1],
        label_classes=label_classes,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        extra=getattr(manager, "eval_metrics", None),
    )
    manager.save_metadata(meta)
    print("✔Training complete. Metadata saved.")

    # print the evaluation metrics if available
    if hasattr(manager, "eval_metrics") and manager.eval_metrics:
        print("Final evaluation metrics:", manager.eval_metrics)

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
            out_dir = os.path.join(cfg["root_dir"], "model")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{cfg.get("label", "")}_val_eval.json'.strip("_"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"[INFO] Saved validation evaluation → {out_path}")

def main():
    p = argparse.ArgumentParser(description="Train an EMG gesture classification model.")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--root_dir",   type=str, required=True)
    p.add_argument("--label",      type=str, default="")
    p.add_argument("--kfold",      action="store_true")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--save_eval", action="store_true",
                   help="After training, save the validation-set predictions/metrics to JSON for plotting.")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "label": args.label or cfg.get("label", ""),
        "kfold": args.kfold or cfg.get("kfold", False),
        "overwrite": args.overwrite or cfg.get("overwrite", False),
        "verbose": args.verbose or cfg.get("verbose", False),
    })

    train_model(cfg, args.save_eval)


if __name__ == "__main__":
    main()
