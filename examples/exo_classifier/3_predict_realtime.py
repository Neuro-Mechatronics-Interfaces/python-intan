#!/usr/bin/env python3
"""
3_predict_realtime.py

Real-time EMG gesture classification from Intan RHX device using paper-style processing.

Key differences from standard approach:
- 120 Hz high-pass filter (4th-order Butterworth) instead of envelope
- Flexible feature selection (default: RMS-only like paper)
- PCA dimensionality reduction to K=30
- Paper-style neural network (30→512→512→N)
- 250ms non-overlapping windows

This matches the preprocessing pipeline in 1_build_training_dataset.py
and the model architecture in 2_train_model.py

Examples:
---------
# Paper-style: RMS only (default)
python 3_predict_realtime.py \
    --root_dir "./trained_model" \
    --label "exo_gestures" \
    --verbose

# Use multiple features
python 3_predict_realtime.py \
    --root_dir "./trained_model" \
    --label "exo_gestures" \
    --features root_mean_square variance waveform_length \
    --verbose

# With LSL output and majority vote smoothing
python 3_predict_realtime.py \
    --root_dir "./trained_model" \
    --label "exo_gestures" \
    --infer_hz 10 \
    --smooth_k 5 \
    --use_lsl \
    --verbose

# All 7 features (like standard approach)
python 3_predict_realtime.py \
    --root_dir "./trained_model" \
    --label "exo_gestures" \
    --features root_mean_square variance waveform_length zero_crossings \
               slope_sign_changes mean_absolute_value integrated_emg \
    --verbose
"""

import re
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import deque, defaultdict
from scipy import signal
import joblib
import json
import os

from intan.interface import IntanRHXDevice, LSLMessagePublisher
from intan.processing import EMGPreprocessor
from intan.io import (
    load_metadata_json,
)

# Import quality control helpers
try:
    from channel_quality_helpers import initialize_realtime_qc, get_good_channels_realtime

    QC_AVAILABLE = True
except ImportError:
    logging.warning("channel_quality_helpers not available - quality control disabled")
    QC_AVAILABLE = False

CHAN_RE = re.compile(r'^\s*([A-Da-d])\s*[-_ ]?\s*(\d{1,3})\s*$')


class PaperStyleEMGClassifier(nn.Module):
    """
    Neural network architecture matching the published paper:
    Input: 30 dimensions (after PCA)
    Hidden layer 1: 512 nodes + Dropout(0.2)
    Hidden layer 2: 512 nodes + Dropout(0.2)
    Output: N classes (softmax via CrossEntropyLoss)
    """

    def __init__(self, input_dim=30, output_dim=10):
        super(PaperStyleEMGClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, output_dim)
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


def normalize_to_1based(names):
    """
    Accepts names like 'B-000' or 'B-0' or 'b_17' and returns canonical 1-based 'B-###'.
    If any '000' is seen, we treat the set as zero-indexed and shift +1.
    """
    parsed = []
    saw_zero = False
    for nm in names:
        m = CHAN_RE.match(str(nm))
        if not m:
            continue
        port = m.group(1).upper()
        num = int(m.group(2))
        if num == 0:
            saw_zero = True
        parsed.append((port, num))

    # If any zero present, interpret the whole set as zero-indexed and shift +1
    shift = 1 if saw_zero else 0

    canon = []
    for port, num in parsed:
        n1 = num + shift
        canon.append(f"{port}-{n1:03d}")
    return canon, bool(saw_zero)


def enable_trained_channels(dev, trained_names):
    """
    Enable Intan wideband channels based on names like 'A-003'.
    Returns dict { 'a': [idx0,...], 'b': [...], ... } of 0-based indices enabled per port.
    """
    by_port = defaultdict(list)

    logging.debug(f"Trained channel names to enable: {trained_names}")

    for nm in trained_names:
        m = CHAN_RE.match(str(nm))
        if not m:
            logging.warning(f"Skip unrecognized channel name: {nm!r}")
            continue

        port = m.group(1).lower()  # 'a' / 'b' / 'c' / 'd'
        ch_1b = int(m.group(2))  # e.g. 3
        idx0 = ch_1b - 1  # 0-based index
        if 0 <= idx0 < 128:
            by_port[port].append(idx0)
        else:
            logging.warning(f"Out-of-range channel {nm} -> {port.upper()}-{ch_1b:03d}")

    # Dedup, sort, and enable per port
    for port, idxs in by_port.items():
        idxs = sorted(set(idxs))
        try:
            dev.enable_wide_channel(idxs, port=port)  # vector form, if supported
        except Exception:
            for i in idxs:
                dev.enable_wide_channel(i, port=port)  # scalar fallback
        by_port[port] = idxs

    return by_port


def select_training_channels_by_name(raw_data: np.ndarray, device_names: list, trained_names: list):
    """
    Reorder channels from device order to training order.

    Parameters
    ----------
    raw_data : np.ndarray
        Data in device order, shape (n_device_channels, n_samples)
    device_names : list
        Channel names in device order (e.g., ['A-001', 'A-002', ...])
    trained_names : list
        Channel names in training order (e.g., ['B-010', 'A-001', ...])

    Returns
    -------
    reordered : np.ndarray
        Data in training order, shape (n_trained_channels, n_samples)
    missing : list
        Names that couldn't be found
    """
    # Create mapping from name to index in device data
    device_map = {name: idx for idx, name in enumerate(device_names)}

    n_samples = raw_data.shape[1] if raw_data.ndim > 1 else len(raw_data)
    reordered = []
    missing = []

    for name in trained_names:
        if name in device_map:
            idx = device_map[name]
            reordered.append(raw_data[idx, :] if raw_data.ndim > 1 else raw_data[idx])
        else:
            logging.warning(f"Channel {name} not found in device data")
            missing.append(name)
            # Append zeros for missing channel
            reordered.append(np.zeros(n_samples))

    return np.array(reordered), missing


def load_paper_style_model(root_dir: str, label: str, device: torch.device):
    """
    Load paper-style model artifacts.

    Returns
    -------
    model : PaperStyleEMGClassifier
        Loaded neural network model
    scaler : StandardScaler
        Feature scaler
    pca : PCA
        PCA transformer
    label_encoder : LabelEncoder
        Label encoder
    metadata : dict
        Model metadata
    """
    model_dir = os.path.join(root_dir, "model")
    prefix = f"{label}_" if label else ""

    # Load metadata
    metadata_path = os.path.join(model_dir, f"{prefix}metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load preprocessing artifacts
    scaler_path = os.path.join(model_dir, f"{prefix}scaler.pkl")
    pca_path = os.path.join(model_dir, f"{prefix}pca.pkl")
    encoder_path = os.path.join(model_dir, f"{prefix}label_encoder.pkl")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    label_encoder = joblib.load(encoder_path)

    # Load model
    model_path = os.path.join(model_dir, f"{prefix}model.pth")

    input_dim = metadata['model']['input_dim']
    output_dim = metadata['model']['output_dim']

    model = PaperStyleEMGClassifier(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logging.info(f"Loaded paper-style model: {input_dim}→512→512→{output_dim}")
    logging.info(f"Classes: {label_encoder.classes_.tolist()}")

    return model, scaler, pca, label_encoder, metadata


def predict_with_confidence(model, scaler, pca, label_encoder, features, device):
    """
    Run inference and return prediction with confidence.

    Parameters
    ----------
    model : PaperStyleEMGClassifier
        Neural network model
    scaler : StandardScaler
        Feature scaler
    pca : PCA
        PCA transformer
    label_encoder : LabelEncoder
        Label encoder
    features : np.ndarray
        RMS features, shape (n_channels,)
    device : torch.device
        PyTorch device

    Returns
    -------
    label : str
        Predicted class label
    confidence : float
        Prediction confidence (softmax probability)
    """
    # Apply z-score normalization
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Apply PCA
    features_pca = pca.transform(features_scaled)

    # Run inference
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_pca).to(device)
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    # Decode label
    label = label_encoder.inverse_transform([pred_idx])[0]

    return label, confidence


def run(
        root_dir: str,
        label: str = "",
        infer_ms: int | None = None,
        infer_hz: float | None = None,
        seconds_total: float = 0.0,
        smooth_k: int = 5,
        use_lsl: bool = False,
        feature_fns: list[str] | None = None,
        print_all: bool = False,
        confidence_threshold: float = 0.0,
        enable_qc: bool = False,
        verbose: bool = False,
):
    """
    Stream EMG from Intan RHX device and run paper-style gesture inference.
    """

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    logging.info("=" * 70)
    logging.info("PAPER-STYLE REAL-TIME PREDICTION")
    logging.info("=" * 70)

    # Load metadata
    try:
        meta = load_metadata_json(root_dir, label=label)
    except Exception:
        # Fallback: try to load from model directory
        model_dir = os.path.join(root_dir, "model")
        prefix = f"{label}_" if label else ""
        metadata_path = os.path.join(model_dir, f"{prefix}metadata.json")
        with open(metadata_path, "r") as f:
            meta = json.load(f)

    data_meta = meta.get("data", meta)

    # Get parameters from metadata
    window_ms = int(data_meta.get("window_ms", 250))
    step_ms = int(data_meta.get("step_ms", 250))
    sample_rate = float(data_meta.get("sample_rate_hz", 4000.0))

    logging.info(f"Training parameters: window={window_ms}ms, step={step_ms}ms, fs={sample_rate}Hz")

    # Calculate inference period
    if infer_ms is not None:
        infer_period_s = max(1e-3, infer_ms / 1000.0)
    elif infer_hz is not None and infer_hz > 0:
        infer_period_s = max(1e-3, 1.0 / float(infer_hz))
    else:
        # Default: match training step
        infer_period_s = max(1e-3, step_ms / 1000.0)

    logging.info(f"Inference period: {infer_period_s * 1000:.1f} ms ({1.0 / infer_period_s:.1f} Hz)")

    # Get training channel names
    trained_names_raw = data_meta.get("selected_channel_names") or data_meta.get("channel_names")

    if not trained_names_raw:
        # Fallback: try to load from dataset file
        try:
            dataset_path_fallback = os.path.join(root_dir, f"{label}_training_dataset.npz")
            if os.path.exists(dataset_path_fallback):
                dataset = np.load(dataset_path_fallback, allow_pickle=True)
                if "channel_names" in dataset:
                    trained_names_raw = dataset["channel_names"].tolist()
                    logging.info(f"Loaded channel names from dataset file")
        except Exception as e:
            logging.warning(f"Could not load channel names from dataset: {e}")

    if not trained_names_raw:
        raise RuntimeError(
            "Cannot find channel names in metadata. "
            "Please retrain your model with the updated training script that saves channel names."
        )

    trained_names_raw = [str(x).strip() for x in trained_names_raw]
    logging.info(f"Training channel names (raw): {trained_names_raw[:5]}... ({len(trained_names_raw)} total)")

    trained_names, was_zero = normalize_to_1based(trained_names_raw)
    if was_zero:
        logging.info("Detected 0-based trained names; normalized to 1-based (X-###)")

    logging.info(f"Training channels: {len(trained_names)}")

    # Setup device
    logging.info("\nConnecting to Intan RHX device...")
    dev = IntanRHXDevice()
    fs = float(getattr(dev, "sample_rate", sample_rate))
    logging.info(f"Device sample rate: {fs} Hz")

    if abs(fs - sample_rate) > 1.0:
        logging.warning(f"Device sample rate ({fs} Hz) != training sample rate ({sample_rate} Hz)")

    dev.clear_all_data_outputs()

    # Enable trained channels
    enabled = enable_trained_channels(dev, trained_names)

    # Build device channel order (deterministic: sorted by port, then by index)
    device_active_names = []
    for port in sorted(enabled.keys()):
        for idx0 in enabled[port]:
            device_active_names.append(f"{port.upper()}-{idx0 + 1:03d}")

    n_enabled = len(device_active_names)
    logging.info(f"Enabled {n_enabled} channels: " + ", ".join(
        f"{p.upper()}:{len(idxs)}" for p, idxs in sorted(enabled.items())
    ))

    # Update device channel count
    if getattr(dev, "num_channels", None) != n_enabled:
        dev.num_channels = n_enabled
        if hasattr(dev, "_update_read_size"):
            dev._update_read_size()

    # Start streaming
    dev.start_streaming()
    logging.info("✓ Device streaming started")

    # Load paper-style model
    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nLoading paper-style model (device: {device_torch})...")
    model, scaler, pca, label_encoder, model_meta = load_paper_style_model(root_dir, label, device_torch)

    # Initialize preprocessor (using your existing class)
    # Uses 120 Hz HPF, no envelope (envelope_cutoff=None)
    preprocessor = EMGPreprocessor(
        fs=fs,
        envelope_cutoff=None,  # Paper-style: no envelope, just HPF
        verbose=verbose
    )

    # Define feature functions to use (comment out ones you don't want)
    if feature_fns is None:
        feature_fns = [
            'root_mean_square',  # Paper uses RMS only
            # 'variance',             # Uncomment to add variance
            # 'waveform_length',      # Uncomment to add WL
            # 'zero_crossings',       # Uncomment to add ZC
            # 'slope_sign_changes',   # Uncomment to add SSC
            # 'mean_absolute_value',  # Uncomment to add MAV
            # 'integrated_emg',       # Uncomment to add IEMG
        ]
    logging.info(f"Feature functions: {feature_fns}")

    # Calculate window size in samples
    window_samples = int(window_ms * fs / 1000.0)
    logging.info(f"Window size: {window_samples} samples ({window_ms} ms)")

    # Calculate expected feature dimension
    n_expected = len(scaler.mean_)  # Should match n_channels × n_features_per_channel
    logging.info(f"Expected feature dimension: {n_expected} (from scaler)")

    # ========================================================================
    # INITIALIZE CHANNEL QUALITY CONTROL (optional)
    # ========================================================================
    qc = None
    if enable_qc:
        if QC_AVAILABLE:
            logging.info("Initializing channel quality control...")
            qc = initialize_realtime_qc(fs=fs, n_channels=len(trained_names), window_sec=0.5)
            if qc:
                logging.info("✓ Quality control enabled")
            else:
                logging.warning("Quality control initialization failed")
        else:
            logging.warning("Quality control requested but unavailable")
    # ========================================================================

    # Optional LSL publisher
    lsl = None
    if use_lsl:
        try:
            lsl = LSLMessagePublisher(
                name="EMGGesture",
                stream_type="Markers",
                only_on_change=False
            )
            logging.info("✓ LSL publisher initialized")
        except Exception as e:
            logging.warning(f"Could not initialize LSL: {e}")

    # Majority vote smoother
    recent = deque(maxlen=max(1, smooth_k))
    last_print_label = None

    logging.info("\n" + "=" * 70)
    logging.info("STREAMING... press Ctrl+C to stop")
    logging.info("=" * 70)

    t0 = time.monotonic()
    next_t = t0

    try:
        while True:
            if seconds_total > 0 and (time.monotonic() - t0) >= seconds_total:
                break

            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)

            # Get latest window from device
            try:
                raw_win = dev.get_latest_window(window_ms)  # (C, win_samples)

                if verbose:
                    logging.debug(f"Got raw window: shape={raw_win.shape}, expected=({n_enabled}, {window_samples})")

            except ValueError as e:
                # Buffer not yet filled
                if verbose:
                    logging.debug(f"Buffer not ready: {e}")
                next_t += infer_period_s
                continue

            # Reorder channels to match training order
            emg_win, missing = select_training_channels_by_name(
                raw_win, device_active_names, trained_names
            )

            if verbose:
                logging.debug(
                    f"Reordered window: shape={emg_win.shape}, expected=({len(trained_names)}, {window_samples})")

            if missing and verbose:
                logging.debug(f"Missing channels: {missing}")

            # ====================================================================
            # CHANNEL QUALITY CONTROL (optional)
            # ====================================================================
            if qc is not None:
                good_channels, bad_channels = get_good_channels_realtime(qc, emg_win, verbose=verbose)

                # Zero out bad channels (keeps feature dimension consistent)
                if bad_channels:
                    emg_win_qc = emg_win.copy()
                    for ch in bad_channels:
                        emg_win_qc[ch, :] = 0.0  # Zero out bad channels
                    emg_win = emg_win_qc

                    if verbose:
                        logging.debug(f"QC: Zeroed {len(bad_channels)} bad channels")

                    # Optional: Skip prediction if too many bad channels
                    if len(bad_channels) > 0.5 * len(trained_names):  # >50% bad
                        logging.warning(f"Skipping prediction: {len(bad_channels)}/{len(trained_names)} channels bad")
                        next_t += infer_period_s
                        continue
            # ====================================================================

            # Apply preprocessing (120 Hz HPF via your EMGPreprocessor)
            emg_filtered = preprocessor.preprocess(emg_win)

            if verbose:
                logging.debug(f"Filtered window: shape={emg_filtered.shape}")

            # Extract features using your extract_emg_features method
            # This returns shape (n_windows, n_features) but we want just the latest window
            try:
                features_matrix = preprocessor.extract_emg_features(
                    emg_filtered,
                    window_ms=window_ms,
                    step_ms=window_ms,  # Non-overlapping, extract single window
                    feature_fns=feature_fns,
                    progress=False
                )

                if verbose:
                    logging.debug(f"Feature matrix: shape={features_matrix.shape}, expected=(1 or more, {n_expected})")

                # Take the last (most recent) window
                if features_matrix.ndim == 1:
                    features = features_matrix
                else:
                    features = features_matrix[-1, :]  # Get last window

                if verbose:
                    logging.debug(
                        f"Features extracted: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

            except (ValueError, IndexError) as e:
                if verbose:
                    logging.debug(f"Feature extraction failed: {e}")
                next_t += infer_period_s
                continue

            # Check feature dimension
            n_expected = len(scaler.mean_)
            if features.shape[0] != n_expected:
                logging.warning(
                    f"Feature dim {features.shape[0]} != scaler expectation {n_expected} (skipping)"
                )
                next_t += infer_period_s
                continue

            # Run inference
            pred_label, confidence = predict_with_confidence(
                model, scaler, pca, label_encoder, features, device_torch
            )

            # Add to recent predictions
            recent.append(pred_label)

            # Publish to LSL
            if lsl is not None:
                lsl.publish(pred_label)

            # Majority vote smoothing
            if recent:
                vals, counts = np.unique(recent, return_counts=True)
                smoothed = vals[np.argmax(counts)]
            else:
                smoothed = pred_label

            # Print predictions based on mode
            elapsed = time.monotonic() - t0

            # Determine if we should print this prediction
            should_print = False

            if print_all:
                # Print everything
                should_print = True
            elif confidence >= confidence_threshold and smoothed != last_print_label:
                # Print if confidence exceeds threshold AND label changed
                should_print = True
            elif confidence_threshold == 0.0 and smoothed != last_print_label:
                # Default behavior: print on change (no confidence filter)
                should_print = True

            if should_print:
                # Color code by confidence
                conf_marker = "✓" if confidence >= 0.8 else "~" if confidence >= 0.5 else "?"

                logging.info(
                    f"[{elapsed:6.2f}s] {conf_marker} pred={pred_label} (conf={confidence:.2f})  "
                    f"smoothed={smoothed} (k={len(recent)})"
                )
                last_print_label = smoothed

            # Schedule next inference
            next_t += infer_period_s

            # Catch up if we fell behind
            if (time.monotonic() - next_t) > 2 * infer_period_s:
                next_t = time.monotonic() + infer_period_s

    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")

    finally:
        try:
            dev.stop_streaming()
            dev.set_run_mode("stop")
        except Exception:
            pass

        if lsl is not None:
            lsl.close()

        dev.close()
        logging.info("Streaming stopped and device closed")


def main():
    p = argparse.ArgumentParser(
        description="Real-time EMG gesture prediction using paper-style processing"
    )
    p.add_argument("--root_dir", type=str, required=True,
                   help="Folder with trained model/metadata")
    p.add_argument("--label", type=str, default="",
                   help="Model label prefix")
    p.add_argument("--infer_ms", type=int, default=None,
                   help="Run inference every N ms (overrides infer_hz)")
    p.add_argument("--infer_hz", type=float, default=None,
                   help="Run inference N times per second")
    p.add_argument("--seconds", type=float, default=0,
                   help="Total seconds to run (0 for infinite)")
    p.add_argument("--smooth_k", type=int, default=5,
                   help="Majority-vote window size (0/1 disables)")
    p.add_argument("--use_lsl", action="store_true",
                   help="Send predictions to LSL 'Markers' stream")
    p.add_argument("--features", type=str, nargs='+', default=None,
                   help="Feature functions to use (default: root_mean_square). "
                        "Options: root_mean_square, variance, waveform_length, "
                        "zero_crossings, slope_sign_changes, mean_absolute_value, integrated_emg")
    p.add_argument("--print_all", action="store_true",
                   help="Print every prediction (default: only print on change)")
    p.add_argument("--confidence_threshold", type=float, default=0.0,
                   help="Only print predictions with confidence >= threshold (0.0-1.0). "
                        "Use with --print_all or will still only print on label change. "
                        "Example: 0.5 = only show predictions with 50%+ confidence")
    p.add_argument("--enable_qc", action="store_true",
                   help="Enable channel quality control (zeros out bad channels in realtime)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose debug output")

    args = p.parse_args()

    run(
        root_dir=args.root_dir,
        label=args.label,
        infer_ms=args.infer_ms,
        infer_hz=args.infer_hz,
        seconds_total=args.seconds,
        smooth_k=args.smooth_k,
        use_lsl=args.use_lsl,
        feature_fns=args.features,
        print_all=args.print_all,
        confidence_threshold=args.confidence_threshold,
        enable_qc=args.enable_qc,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
