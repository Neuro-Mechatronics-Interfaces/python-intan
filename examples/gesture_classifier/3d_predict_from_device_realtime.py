#!/usr/bin/env python3
"""
3c_predict_streaming_from_device.py

Stream EMG from Intan RHX TCP client and run gesture inference at a user-chosen rate.
- Locks window/step/envelope + channel order to training metadata
- Uses device circular buffer (no blocking)
- Optional majority-vote smoothing
- Optional LSL publishing

Examples:
  python 3c_predict_streaming_from_device.py \
    --root_dir "G:/.../2024_11_11" --label "128_channels" --infer_ms 100 --smooth_k 5 --use_lsl --verbose

  python 3c_predict_streaming_from_device.py \
    --root_dir "G:/.../2024_11_11" --label "128_channels" --infer_hz 20
"""
import re
import time
import argparse
import logging
import numpy as np
from collections import deque, defaultdict

from intan.interface import IntanRHXDevice, LSLMessagePublisher
from intan.processing import EMGPreprocessor
from intan.ml import ModelManager, EMGClassifier

from intan.io import (
    get_trained_channel_names,
    load_metadata_json,
    lock_params_to_meta,
    select_training_channels_by_name,
)

CHAN_RE = re.compile(r'^\s*([A-Da-d])\s*[-_ ]?\s*(\d{1,3})\s*$')

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

    print(f"Trained names: {trained_names}")


    for nm in trained_names:
        m = CHAN_RE.match(str(nm))
        if not m:
            logging.warning(f"skip unrecognized channel name: {nm!r}")
            continue

        print("Matched ", nm)
        port = m.group(1).lower()             # 'a' / 'b' / 'c' / 'd'
        ch_1b = int(m.group(2))               # e.g. 3
        idx0  = ch_1b - 1                     # 0-based index
        if 0 <= idx0 < 128:
            by_port[port].append(idx0)
        else:
            logging.warning(f"out-of-range channel {nm} -> {port.upper()}-{ch_1b:03d}")

    # Dedup, sort, and enable per port
    for port, idxs in by_port.items():
        idxs = sorted(set(idxs))
        try:
            dev.enable_wide_channel(idxs, port=port)   # vector form, if supported
        except Exception:
            for i in idxs:
                dev.enable_wide_channel(i, port=port)  # scalar fallback
        by_port[port] = idxs

    return by_port

def _get_device_channel_names(dev) -> list[str]:
    if hasattr(dev, "get_channel_names"):
        try:
            names = list(dev.get_channel_names())
            if names:
                return names
        except Exception:
            pass
    return [f"CH{i}" for i in range(getattr(dev, "num_channels", 128))]

def _top1_with_conf(manager, Xrow):
    if hasattr(manager, "predict_proba"):
        try:
            proba = manager.predict_proba(Xrow)[0]
            idx = int(np.argmax(proba))
            label = (
                manager.inverse_transform([idx])[0]
                if hasattr(manager, "inverse_transform")
                else manager.predict(Xrow)[0]
            )
            return label, float(proba[idx])
        except Exception:
            pass
    return manager.predict(Xrow)[0], None

def _compute_infer_period_s(infer_ms: int | None, infer_hz: float | None, fallback_step_ms: int) -> float:
    if infer_ms is not None:
        return max(1e-3, infer_ms / 1000.0)
    if infer_hz is not None and infer_hz > 0:
        return max(1e-3, 1.0 / float(infer_hz))
    # default: lock to training step
    return max(1e-3, fallback_step_ms / 1000.0)

def _training_names_from_meta(root_dir: str, label: str, meta: dict, data_meta: dict) -> list[str]:
    # Prefer the names saved with the training subset/order
    for key in ("selected_channel_names", "selected_channels_names", "channel_names"):
        names = data_meta.get(key) or meta.get(key)
        if isinstance(names, (list, tuple)) and names:
            return list(names)
    try:
        names = get_trained_channel_names(root_dir, label=label)
        if names:
            return list(names)
    except Exception:
        pass
    raise RuntimeError

def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None,
    infer_ms: int | None = None, infer_hz: float | None = None, seconds_total: float = 60.0, smooth_k: int = 5,
    use_lsl: bool = False, verbose: bool = False):
    """Stream EMG from Intan RHX device and run gesture inference at a user-chosen rate."""

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # --- Load metadata and lock params like training ---
    meta = load_metadata_json(root_dir, label=label)
    data_meta = meta.get("data", meta)

    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        data_meta, window_ms, step_ms, selected_channels=None
    )
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Inference period (how often you run inference)
    infer_period_s = _compute_infer_period_s(infer_ms, infer_hz, fallback_step_ms=step_ms)
    logging.info(
        f"Inference cadence: every {infer_period_s*1000:.1f} ms "
        f"(window={window_ms} ms, training step={step_ms} ms)"
    )

    # Get training channel names
    trained_names_raw = _training_names_from_meta(root_dir, label, meta, data_meta)
    trained_names_raw = [str(x).strip() for x in trained_names_raw]
    print(f"Trained channel names: {trained_names_raw}")
    trained_names, was_zero = normalize_to_1based(trained_names_raw)
    if was_zero:
        print("[INFO] Detected 0-based trained names; normalized to 1-based (X-###).")
    logging.info(f"Training channel count: {len(trained_names)}")

    # Optional LSL publisher
    lsl = LSLMessagePublisher(name="EMGGesture", stream_type="Markers", only_on_change=False) if use_lsl else None

    # --- Device setup ---
    dev = IntanRHXDevice()
    fs = float(getattr(dev, "sample_rate", data_meta.get("sample_rate_hz", 4000.0)))
    dev.clear_all_data_outputs()
    enabled = enable_trained_channels(dev, trained_names)

    # Build the actual device order we just enabled (to match raw_win channel order)
    device_active_names = []
    for port in sorted(enabled.keys()):  # deterministic ordering by port
        for idx0 in enabled[port]:  # ascending within the port
            device_active_names.append(f"{port.upper()}-{idx0 + 1:03d}")

    logging.info("Enabled channels: " + ", ".join(
        f"{p.upper()}:{len(is_)}" for p, is_ in sorted(enabled.items())
    ))

    # make the device reflect the enabled channel count
    n_enabled = len(device_active_names)  # 116 in your case
    if getattr(dev, "num_channels", None) != n_enabled:
        dev.num_channels = n_enabled
        # make sure socket read sizes match the new channel count
        if hasattr(dev, "_update_read_size"):
            dev._update_read_size()

    dev.start_streaming()

    # --- Model & preprocessing reused at each tick ---
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_expected = len(manager.scaler.mean_)

    try:
        pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    except TypeError:
        pre = EMGPreprocessor(fs=fs, env_cut=env_cut)

    # Majority vote smoother
    recent = deque(maxlen=max(1, smooth_k))
    last_print_label = None

    logging.info("Streaming… press Ctrl+C to stop.")
    t0 = time.monotonic()
    next_t = t0

    try:
        while True:
            if seconds_total and (time.monotonic() - t0) >= seconds_total:
                break

            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)

            # Make sure buffer has at least one window worth of samples
            try:
                raw_win = dev.get_latest_window(window_ms)  # (C, win_samples)
            except ValueError:
                # buffer not yet filled to window size; wait a tick
                next_t += infer_period_s
                continue

            # Reorder to training order
            emg_win, _ = select_training_channels_by_name(raw_win, device_active_names, trained_names)

            # Preprocess and extract exactly one feature vector using window_ms as hop
            emg_pp = pre.preprocess(emg_win)
            Xw = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=window_ms, progress=False)

            if Xw.ndim == 1:
                Xw = Xw.reshape(1, -1)

            if Xw.shape[1] != n_expected:
                logging.warning(f"Feature dim {Xw.shape[1]} != scaler expectation {n_expected} (skipping tick)")
            else:
                pred_label, conf = _top1_with_conf(manager, Xw)
                recent.append(pred_label)

                # LSL publish (optional)
                if lsl is not None:
                    lsl.publish(pred_label)

                # Majority vote smoothing
                if recent:
                    vals, counts = np.unique(recent, return_counts=True)
                    smoothed = vals[np.argmax(counts)]
                else:
                    smoothed = pred_label

                if smoothed != last_print_label:
                    if conf is None:
                        logging.info(f"[{(time.monotonic()-t0):6.2f}s] pred={pred_label}  smoothed={smoothed} (k={len(recent)})")
                    else:
                        logging.info(f"[{(time.monotonic()-t0):6.2f}s] pred={pred_label} (p={conf:.2f})  smoothed={smoothed} (k={len(recent)})")
                    last_print_label = smoothed

            # schedule next tick; catch up if we fell behind
            next_t += infer_period_s
            if (time.monotonic() - next_t) > 2 * infer_period_s:
                next_t = time.monotonic() + infer_period_s

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    finally:
        try:
            dev.stop_streaming()
            dev.set_run_mode("stop")
        except Exception:
            pass
        if lsl is not None:
            lsl.close()
        dev.close()
        logging.info("Streaming stopped and device closed.")

def main():
    p = argparse.ArgumentParser("Streaming EMG gesture prediction from Intan device (training-locked).")
    p.add_argument("--root_dir",    type=str, required=True, help="Folder with trained model/metadata.")
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None, help="Override; else from metadata.")
    p.add_argument("--step_ms",     type=int, default=None, help="Override; else from metadata.")
    p.add_argument("--infer_ms",    type=int, default=None, help="Run inference every N ms (overrides infer_hz).")
    p.add_argument("--infer_hz",    type=float, default=None, help="Run inference N times per second.")
    p.add_argument("--seconds",     type=float, default=0, help="Total seconds to run (<=0 for infinite).")
    p.add_argument("--smooth_k",    type=int, default=5, help="Majority-vote window (0/1 disables).")
    p.add_argument("--use_lsl",     action="store_true", help="Send predictions to an LSL 'Markers' stream.")
    p.add_argument("--verbose",     action="store_true", help="Enable verbose debug.")
    args = p.parse_args()

    run(
        root_dir=args.root_dir,
        label=args.label,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        infer_ms=args.infer_ms,
        infer_hz=args.infer_hz,
        seconds_total=args.seconds,
        smooth_k=args.smooth_k,
        use_lsl=args.use_lsl,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
