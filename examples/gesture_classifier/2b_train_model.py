#!/usr/bin/env python3
"""
2_train_model.py

Train an EMG gesture classifier from a pre-built dataset (.npz).
- Default: loads a single dataset from --root_dir (your old behavior)
- NEW: --train_npz [one or more] and optional --test_npz for external test
- NEW: --save_tag to suffix saved artifacts (prevents overwrites)
"""

import os
import json
import argparse
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from intan.io import load_config_file
from intan.ml import ModelManager, EMGClassifier


# -------------------------
# Helpers (NEW / unchanged)
# -------------------------

def load_npz_list(npz_paths):
    Xs, ys, metas = [], [], []
    for p in npz_paths:
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
            "emg_fs": float(d["emg_fs"]) if "emg_fs" in d.files else None,  # NEW
        })
    fdim = {x.shape[1] for x in Xs}
    if len(fdim) != 1:
        raise ValueError(f"Feature dims differ across inputs: {fdim} for {npz_paths}")
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y, metas

def _resolve_paths(paths, root_dir):
    out = []
    for p in (paths or []):
        out.append(p if os.path.isabs(p) or os.path.exists(p) else os.path.join(root_dir, p))
    return out


def _find_dataset_path(root_dir: str, label: str | None) -> str:
    """
    Try common dataset filenames. Prefer label-specific, then generic.
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
    return candidates[0] if candidates else os.path.join(root_dir, "training_dataset.npz")


def _safe_json_item(x):
    try:
        return x.item()
    except Exception:
        return x


def _load_single_dataset(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as d:
        X = d["X"]
        y = d["y"] if "y" in d.files else d["y_id"]
        meta = {
            "emg_fs": float(d["emg_fs"]) if "emg_fs" in d.files else None,
            "window_ms": int(d["window_ms"]) if "window_ms" in d.files else None,
            "step_ms": int(d["step_ms"]) if "step_ms" in d.files else None,
            "feature_spec": json.loads(str(_safe_json_item(d["feature_spec"]))) if "feature_spec" in d.files else None,
            "selected_channels": d["selected_channels"].tolist() if "selected_channels" in d.files else [],
            "channel_names": d["channel_names"].tolist() if "channel_names" in d.files else [],
            "class_names": d["class_names"].tolist() if "class_names" in d.files else None,
        }
    return X, y, meta


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -------------------------
# Core training flow
# -------------------------

def train_model(cfg: dict, save_eval: bool = False):
    label    = cfg.get("label", "")
    root_dir = cfg["root_dir"]
    save_tag = cfg.get("save_tag", "").strip()

    # Initialize manager with save_tag baked into filenames (if supported via config)
    # The ModelManager in your code saves to <root>/model/...; we’ll keep that.
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config=cfg)

    use_external = False
    X_test = y_test = None
    if cfg.get("train_npz"):  # NEW path
        #train_list = cfg["train_npz"]
        train_list = _resolve_paths(cfg["train_npz"], root_dir)
        logging.info(f"[trainer] Loading train NPZ: {train_list}")
        X_train, y_train, metas = load_npz_list(train_list)

        # if a test set was provided, load it
        if cfg.get("test_npz"):
            tpath = cfg["test_npz"]
            logging.info(f"[trainer] Loading external TEST NPZ: {tpath}")
            with np.load(tpath, allow_pickle=True) as d:
                X_test = d["X"]
                y_test = d["y"] if "y" in d.files else d["y_id"]
                # shape sanity
                if X_test.shape[1] != X_train.shape[1]:
                    raise ValueError(f"Train/Test feature dims differ: {X_train.shape[1]} vs {X_test.shape[1]}")
                meta_test = {
                    "class_names": d["class_names"].tolist() if "class_names" in d.files else None,
                    "window_ms": int(d["window_ms"]) if "window_ms" in d.files else None,
                    "step_ms": int(d["step_ms"]) if "step_ms" in d.files else None,
                }
            use_external = True

        # crude meta from first file (only for metadata snapshot)
        first = metas[0] if metas else {}
        meta_train = {
            "emg_fs": first.get("emg_fs"),
            "window_ms": first.get("window_ms"),
            "step_ms": first.get("step_ms"),
            "feature_spec": None,
            "selected_channels": [],
            "channel_names": [],
            "class_names": metas[0].get("class_names") if metas and metas[0].get("class_names") else None,
        }

    else:
        # Legacy single-dataset path (unchanged)
        npz_path = _find_dataset_path(root_dir, label)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found. Looked for: {npz_path}")
        X_train, y_train, meta_train = _load_single_dataset(npz_path)

    # -------------
    # Train / load
    # -------------

    if (not os.path.isfile(manager.model_path)) or cfg.get("overwrite", False):
        print("Training new model...")
        # When use_external=True we want to train on ALL training data (no internal split beyond what your manager does).
        # Your ModelManager already handles val split & early stopping; we keep it as-is.
        manager.train(X_train, y_train)
    else:
        print("Model exists; loading.")
        manager.load_model()

    # -------------
    # Metadata save
    # -------------
    # Label class names
    if hasattr(manager, "label_encoder"):
        label_classes = [str(c) for c in manager.label_encoder.classes_]
    elif hasattr(manager, "classes_"):
        label_classes = [str(c) for c in manager.classes_]
    else:
        label_classes = sorted([str(c) for c in np.unique(y_train)])

    scaler_mean = getattr(getattr(manager, "scaler", None), "mean_", None)
    scaler_scale = getattr(getattr(manager, "scaler", None), "scale_", None)
    scaler_mean = None if scaler_mean is None else scaler_mean.tolist()
    scaler_scale = None if scaler_scale is None else scaler_scale.tolist()

    meta = manager.build_metadata(
        sample_rate_hz=float(meta_train.get("emg_fs") or 0.0),
        window_ms=meta_train.get("window_ms"),
        step_ms=meta_train.get("step_ms"),
        envelope_cutoff_hz=cfg.get("envelope_cutoff_hz", 5.0),
        selected_channels=meta_train.get("selected_channels", []),
        channel_names=meta_train.get("channel_names", []),
        feature_spec=meta_train.get("feature_spec"),
        n_features=X_train.shape[1],
        label_classes=label_classes,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        extra=getattr(manager, "eval_metrics", None),
    )
    manager.save_metadata(meta)
    print("Training complete. Metadata saved.")

    # Show internal eval (if manager produced it)
    if hasattr(manager, "eval_metrics") and manager.eval_metrics:
        print("Internal/validation metrics:", manager.eval_metrics)

    # -------------------------
    # External test evaluation
    # -------------------------
    if use_external and (X_test is not None):
        print("[INFO] Evaluating on external test set...")
        # manager must expose a predict method; if not, fall back to model
        if hasattr(manager, "predict"):
            y_pred = manager.predict(X_test)
        elif hasattr(manager, "model") and hasattr(manager.model, "predict"):
            y_pred = manager.model.predict(X_test)
        else:
            raise RuntimeError("No predict method available on ModelManager/model.")

        # Ensure arrays of str/object for sklearn
        y_true = np.asarray(y_test, dtype=object)
        y_pred = np.asarray(y_pred, dtype=object)

        labels_sorted = np.unique(np.concatenate([y_true, y_pred]))
        rep_dict = classification_report(
            y_true, y_pred, labels=labels_sorted, zero_division=0, output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

        out_dir = os.path.join(root_dir, "model")
        os.makedirs(out_dir, exist_ok=True)

        # Suffix with save_tag if provided
        tag = f"_{save_tag}" if save_tag else ""
        metrics_path = os.path.join(out_dir, f"{label}_external_metrics{tag}.json".strip("_"))
        payload = {
            "labels": labels_sorted.tolist(),
            "confusion_matrix": cm.tolist(),
            "classification_report": rep_dict,
            "train_npz": cfg.get("train_npz"),
            "test_npz": cfg.get("test_npz"),
        }
        _save_json(payload, metrics_path)
        print(f"[INFO] Saved external test metrics → {metrics_path}")

    # -------------------------
    # Optional: save val preds
    # -------------------------
    if save_eval:
        y_true_val = getattr(manager, "y_val_true", None)
        y_pred_val = getattr(manager, "y_val_pred", None)
        if y_true_val is not None and y_pred_val is not None:
            labels_sorted = np.unique(np.concatenate([y_true_val, y_pred_val]))
            rep_dict = classification_report(
                y_true_val, y_pred_val, labels=labels_sorted, zero_division=0, output_dict=True
            )
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
            _save_json(out, out_path)
            print(f"[INFO] Saved validation evaluation → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Train an EMG gesture classification model.")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--root_dir",   type=str, required=True)
    p.add_argument("--dataset_dir", type=str, default=None)

    # NEW
    p.add_argument("--train_npz", nargs="+", default=None,
                   help="One or more training .npz files (concatenated). If unset, fall back to single dataset in --root_dir.")
    p.add_argument("--test_npz", type=str, default=None,
                   help="Optional external test .npz. If set, we evaluate on it after training.")
    p.add_argument("--save_tag", type=str, default="",
                   help="Suffix for saved artifacts/metrics, e.g., xori_both_np_ps")

    p.add_argument("--label",      type=str, default="", help="Label prefix for model/dataset files.")
    p.add_argument("--kfold",      action="store_true", help="Use k-fold cross-validation instead of train/test split.")
    p.add_argument("--overwrite",  action="store_true", help="Retrain model even if one exists.")
    p.add_argument("--verbose",    action="store_true", help="Enable verbose logging.")
    p.add_argument("--save_eval",  action="store_true",
                   help="Also save internal validation predictions/metrics to JSON (if available).")
    args = p.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)

    # Thread NEW args into cfg so ModelManager can see them if needed
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "dataset_dir": args.dataset_dir or cfg.get("dataset_dir", ""),
        "label": args.label or cfg.get("label", ""),
        "kfold": args.kfold or cfg.get("kfold", False),
        "overwrite": args.overwrite or cfg.get("overwrite", False),
        "verbose": args.verbose or cfg.get("verbose", False),
        "train_npz": args.train_npz,
        "test_npz": args.test_npz,
        "save_tag": args.save_tag,
    })

    train_model(cfg, args.save_eval)


if __name__ == "__main__":
    main()
