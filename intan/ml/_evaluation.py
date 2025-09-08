import os
import logging
from typing import Iterable, Tuple, Optional, Sequence, Mapping
import re
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from intan.io import labels_from_events

def evaluate_against_events(
    file_path: str,
    window_starts: np.ndarray,
    y_pred: Iterable,                                # can be ints or strings
    drop_labels: Tuple[str, ...] = ("Unknown", "Start"),
    *,
    # ---- optional helpers for mapping int preds <-> strings ----
    label_encoder=None,                              # e.g., manager.label_encoder (sklearn)
    class_names: Optional[Sequence[str]] = None,     # alternative to label_encoder
    label_to_id: Optional[Mapping[str, int]] = None, # if you want to map y_true->ids
    # ---- normalization / reporting controls ----
    canonicalize: bool = True,                       # normalize case/spacing/underscores
    alias: Optional[Mapping[str, str]] = None,       # e.g. {'fingersopen':'handopen'}
    zero_division: int = 0,                          # silence undefined metric warnings
    return_metrics: bool = False,                    # optionally return a dict of metrics
    return_arrays: bool = False,                     # also return y_true_c, y_pred_c if True
    verbose: bool = False,
) -> Optional[dict]:
    """
    Print a classification report & confusion matrix using the emg.event file,
    robust to label casing and int/string mismatches.

    If return_metrics=True, returns {'accuracy': float, 'classification_report': dict,
    'confusion_matrix': [[...]] , 'labels': [...]}.
    """
    if verbose:
        logging.info(f"Evaluating against events in: {file_path}")

    if not os.path.isfile(file_path) or window_starts is None or len(window_starts) == 0:
        logging.info("Skipped offline evaluation (no events file or no predictions).")
        return None

    # --- Load true labels aligned to window starts ---
    y_true = labels_from_events(file_path, window_starts)  # array[str]
    y_true = np.asarray(y_true, dtype=object)

    # Drop administrative labels from ground truth (and align preds with same mask)
    mask = ~np.isin(y_true, list(drop_labels))
    y_true = y_true[mask]
    y_pred = np.asarray(list(y_pred))[mask]

    if verbose:
        print(f"Loaded {len(y_true)} true labels from events file, contents: {np.unique(y_true)}")
    if y_true.size == 0:
        logging.warning("No valid windows to evaluate (all Unknown/Start).")
        return None

    # --- Convert predictions to strings if they are integer class IDs ---
    if np.issubdtype(np.asarray(y_pred).dtype, np.integer):
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            classes = np.asarray(label_encoder.classes_, dtype=object)
            y_pred = classes[y_pred.astype(int)]
        elif class_names is not None:
            classes = np.asarray(class_names, dtype=object)
            y_pred = classes[y_pred.astype(int)]
        else:
            if label_to_id is not None:
                y_true_ids = np.array([label_to_id.get(str(s), -1) for s in y_true], dtype=int)
                keep = y_true_ids >= 0
                y_true = y_true_ids[keep]
                y_pred = np.asarray(y_pred, dtype=int)[keep]
            else:
                logging.warning("y_pred are ints but no label mapping provided; results may be misleading.")

    # --- Canonicalize for fair comparison (case/space/underscore-insensitive) ---
    def _canon(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = alias.get(s, s) if alias else s
        if not canonicalize:
            return s
        return re.sub(r"[\s_-]+", "", s.strip()).lower()

    y_true_c = np.array([_canon(s) for s in y_true], dtype=object)
    y_pred_c = np.array([_canon(s) for s in y_pred], dtype=object)

    # --- Guard: align lengths if something went off ---
    n = min(len(y_true_c), len(y_pred_c))
    if n == 0:
        logging.warning("No overlapping samples after preprocessing.")
        return None
    if len(y_true_c) != len(y_pred_c):
        logging.warning(f"Length mismatch (y_true={len(y_true_c)}, y_pred={len(y_pred_c)}), truncating to {n}.")
        y_true_c = y_true_c[:n]
        y_pred_c = y_pred_c[:n]

    # Only report labels that actually appear in either y_true or y_pred
    labels_sorted = np.unique(np.concatenate([y_true_c, y_pred_c]))

    # --- Metrics ---
    acc = accuracy_score(y_true_c, y_pred_c)
    print(f"\nValidation accuracy (canonicalized): {acc:.4f}\n")

    print("=== Classification Report ===")
    rep_text = classification_report_safe(
        y_true_c, y_pred_c, labels=labels_sorted, zero_division=zero_division, output="text"
    )
    print(rep_text)

    print("=== Confusion Matrix ===")
    cm, cm_labels = confusion_matrix_safe(y_true_c, y_pred_c, labels=labels_sorted)
    print(cm)

    if return_metrics:
        rep_dict = classification_report_safe(
            y_true_c, y_pred_c, labels=labels_sorted, zero_division=zero_division, output="dict"
        )
        out = {
            "accuracy": float(acc),
            "classification_report": rep_dict,
            "confusion_matrix": cm.tolist(),
            "labels": list(cm_labels),
        }
        if return_arrays:
            out["y_true_c"] = y_true_c.tolist()
            out["y_pred_c"] = y_pred_c.tolist()
        return out
    return None

def classification_report_safe(
    y_true,
    y_pred,
    *,
    labels=None,
    zero_division: int = 0,
    output: str = "text",   # "text" or "dict"
):
    """
    Wrapper around sklearn.classification_report with robust handling of kwargs
    across sklearn versions, and a simple output switch.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Try with zero_division + output_dict (newer sklearn); gracefully degrade.
    try:
        if output == "dict":
            return classification_report(
                y_true, y_pred, labels=labels, zero_division=zero_division, output_dict=True
            )
        else:
            return classification_report(
                y_true, y_pred, labels=labels, zero_division=zero_division, digits=4
            )
    except TypeError:
        # Older sklearn: no zero_division kw; try again without it.
        if output == "dict":
            try:
                return classification_report(
                    y_true, y_pred, labels=labels, output_dict=True
                )
            except TypeError:
                # Very old: no output_dict either -> return plain text
                return classification_report(y_true, y_pred, labels=labels, digits=4)
        else:
            return classification_report(y_true, y_pred, labels=labels, digits=4)


def confusion_matrix_safe(
    y_true,
    y_pred,
    *,
    labels=None,
    normalize: Optional[str] = None,  # e.g., "true" | "pred" | "all" (sklearn >=0.22)
):
    """
    Wrap sklearn.confusion_matrix; tolerate older versions that may lack normalize kw.
    Returns (cm, labels).
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    except TypeError:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels



# def evaluate_against_events(
#     file_path: str,
#     window_starts: np.ndarray,
#     y_pred: Iterable,                                # can be ints or strings
#     drop_labels: Tuple[str, ...] = ("Unknown", "Start"),
#     *,
#     # ---- optional helpers for mapping int preds <-> strings ----
#     label_encoder=None,                              # e.g., manager.label_encoder (sklearn)
#     class_names: Optional[Sequence[str]] = None,     # alternative to label_encoder
#     label_to_id: Optional[Mapping[str, int]] = None, # if you want to map y_true->ids
#     # ---- normalization / reporting controls ----
#     canonicalize: bool = True,                       # normalize case/spacing/underscores
#     alias: Optional[Mapping[str, str]] = None,       # e.g. {'fingersopen':'handopen'}
#     zero_division: int = 0,                          # silence undefined metric warnings
#     return_metrics: bool = False,                    # optionally return a dict of metrics
#     verbose: bool = False,
# ) -> Optional[dict]:
#     """
#     Print a classification report & confusion matrix using the emg.event file,
#     robust to label casing and int/string mismatches.
#
#     If return_metrics=True, returns {'accuracy': float, 'classification_report': dict,
#     'confusion_matrix': [[...]] , 'labels': [...]}.
#     """
#     if verbose:
#         logging.info(f"Evaluating against events in: {file_path}")
#
#     if not os.path.isfile(file_path) or window_starts is None or len(window_starts) == 0:
#         logging.info("Skipped offline evaluation (no events file or no predictions).")
#         return None
#
#     # --- Load true labels aligned to window starts ---
#     y_true = labels_from_events(file_path, window_starts)  # array[str]
#     y_true = np.asarray(y_true, dtype=object)
#
#     # Drop administrative labels from ground truth (and align preds with same mask)
#     mask = ~np.isin(y_true, list(drop_labels))
#     y_true = y_true[mask]
#     y_pred = np.asarray(list(y_pred))[mask]
#
#     if verbose:
#         print(f"Loaded {len(y_true)} true labels from events file, contents: {np.unique(y_true)}")
#     if y_true.size == 0:
#         logging.warning("No valid windows to evaluate (all Unknown/Start).")
#         return None
#
#     # --- Convert predictions to strings if they are integer class IDs ---
#     if np.issubdtype(np.asarray(y_pred).dtype, np.integer):
#         if label_encoder is not None and hasattr(label_encoder, "classes_"):
#             classes = np.asarray(label_encoder.classes_, dtype=object)
#             y_pred = classes[y_pred.astype(int)]
#         elif class_names is not None:
#             classes = np.asarray(class_names, dtype=object)
#             y_pred = classes[y_pred.astype(int)]
#         else:
#             # As a last resort: keep ints, but try to map y_true -> ids if mapping given
#             if label_to_id is not None:
#                 # map y_true strings -> ints via label_to_id, unknowns get -1 then filtered
#                 y_true_ids = np.array([label_to_id.get(str(s), -1) for s in y_true], dtype=int)
#                 keep = y_true_ids >= 0
#                 y_true = y_true_ids[keep]
#                 y_pred = np.asarray(y_pred, dtype=int)[keep]
#             else:
#                 logging.warning("y_pred are ints but no label mapping provided; results may be misleading.")
#
#     # --- Canonicalize for fair comparison (case/space/underscore-insensitive) ---
#     def _canon(s: str) -> str:
#         if s is None:
#             return ""
#         s = str(s)
#         s = alias.get(s, s) if alias else s
#         if not canonicalize:
#             return s
#         return re.sub(r"[\s_-]+", "", s.strip()).lower()
#
#     y_true_c = np.array([_canon(s) for s in y_true], dtype=object)
#     y_pred_c = np.array([_canon(s) for s in y_pred], dtype=object)
#
#     # --- Guard: align lengths if something went off (shouldn't happen, but safe) ---
#     n = min(len(y_true_c), len(y_pred_c))
#     if n == 0:
#         logging.warning("No overlapping samples after preprocessing.")
#         return None
#     if len(y_true_c) != len(y_pred_c):
#         logging.warning(f"Length mismatch (y_true={len(y_true_c)}, y_pred={len(y_pred_c)}), truncating to {n}.")
#         y_true_c = y_true_c[:n]
#         y_pred_c = y_pred_c[:n]
#
#     # Only report labels that actually appear in either y_true or y_pred
#     labels_sorted = np.unique(np.concatenate([y_true_c, y_pred_c]))
#
#     # --- Metrics ---
#     acc = accuracy_score(y_true_c, y_pred_c)
#     print(f"\nValidation accuracy (canonicalized): {acc:.4f}\n")
#     print("=== Classification Report ===")
#     rep_text = classification_report_safe(y_true_c, y_pred_c, labels=labels_sorted, zero_division=zero_division)
#     print(rep_text)
#     print("=== Confusion Matrix ===")
#     cm = confusion_matrix_safe(y_true_c, y_pred_c, labels=labels_sorted)
#     print(cm)
#
#     if return_metrics:
#         # also return a machine-friendly dict
#         try:
#             from sklearn.metrics import classification_report as _cr
#             rep_dict = _cr(y_true_c, y_pred_c, labels=labels_sorted, zero_division=zero_division, output_dict=True)
#         except TypeError:
#             # older sklearn may not support output_dict in same way; fallback by parsing text is overkill
#             rep_dict = {"report_text": rep_text}
#         return {
#             "accuracy": float(acc),
#             "classification_report": rep_dict,
#             "confusion_matrix": cm.tolist(),
#             "labels": labels_sorted.tolist(),
#         }
#     return None
