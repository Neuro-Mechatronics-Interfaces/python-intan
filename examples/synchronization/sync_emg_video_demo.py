#!/usr/bin/env python3
"""
sync_emg_video_demo.py

Demo showing how to synchronize EMG and video landmark data using
the `intan.processing` synchronization utilities.

Usage (non-interactive):
  python sync_emg_video_demo.py \
      --root_dir /path/to/project \
      --emg_file raw/recording.rhd \
      --landmarks_file media/landmarks/recording_landmarks.npz \
      --output sync/recording_sync.json --plot sync/recording_sync.png

If files are omitted the script will prompt using the repository GUI helpers.
"""

import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


# Allow running from repo without an editable install
try:
    import intan  # noqa: F401
except Exception:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import intan  # noqa: F401
    except Exception:
        print('\n[ERROR] Cannot import `intan`.')
        print('Install the package in editable mode from the repository root:')
        print('  pip install -e .')
        sys.exit(1)

from intan.io import load_single_file, prompt_file
from intan.processing import (
    compute_landmark_movement_signal,
    compute_emg_envelope_signal,
    find_sync_offset,
    save_sync_offset,
)


def normalize_sig(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)


def run_sync(root_dir, emg_file, landmarks_file, out_json=None, out_plot=None, max_offset=10.0):
    # Resolve relative paths
    if not os.path.isabs(emg_file):
        emg_file = os.path.join(root_dir, emg_file)
    if not os.path.isabs(landmarks_file):
        landmarks_file = os.path.join(root_dir, landmarks_file)

    print(f"Loading EMG: {emg_file}")
    data = load_single_file('rhd', emg_file, root_dir, verbose=False)
    emg = data['amplifier_data']  # (channels, samples)
    fs = float(data['frequency_parameters']['amplifier_sample_rate'])

    print(f"Loading landmarks: {landmarks_file}")
    lm = np.load(landmarks_file)
    landmarks = lm['landmarks']
    landmark_times = lm['time_vector']

    # Compute signals
    emg_env, emg_times = compute_emg_envelope_signal(emg, fs)
    lm_signal, lm_times = compute_landmark_movement_signal(landmarks, landmark_times)

    sync = find_sync_offset(emg_env, emg_times, lm_signal, lm_times, max_offset_sec=float(max_offset))
    print(json.dumps(sync, indent=2))

    # Save JSON
    if out_json is None:
        stem = os.path.splitext(os.path.basename(emg_file))[0]
        out_json = os.path.join(root_dir, 'sync', f"{stem}_sync.json")
    save_sync_offset(sync, out_json)
    print(f"Saved sync info to: {out_json}")

    # Optional plot: overlay EMG envelope and (shifted) landmark signal
    if out_plot:
        from scipy.interpolate import interp1d
        lm_t_aligned = lm_times - sync['offset_sec']
        f_lm = interp1d(lm_t_aligned, lm_signal, kind='linear', fill_value=0, bounds_error=False)
        lm_resampled = f_lm(emg_times)

        plt.figure(figsize=(10, 4))
        plt.plot(emg_times, normalize_sig(emg_env), label='EMG envelope', alpha=0.8)
        plt.plot(emg_times, normalize_sig(lm_resampled), label=f'Landmarks (shifted by -{sync["offset_sec"]:.3f}s)', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_plot), exist_ok=True)
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Saved alignment plot to: {out_plot}")

    return sync


def main():
    p = argparse.ArgumentParser(description="Demo: synchronize EMG and video landmarks")
    p.add_argument('--root_dir', default='.', help='Project root directory')
    p.add_argument('--emg_file', help='Path to EMG file (.rhd) relative to root or absolute')
    p.add_argument('--landmarks_file', help='Path to landmarks NPZ relative to root or absolute')
    p.add_argument('--output', help='Output JSON path to save sync info')
    p.add_argument('--plot', help='Output PNG path to save alignment plot')
    p.add_argument('--max_offset', type=float, default=10.0, help='Max offset search window (sec)')
    args = p.parse_args()

    root_dir = args.root_dir or '.'

    emg_file = args.emg_file
    if not emg_file:
        emg_file = prompt_file(title='Select EMG (.rhd) file', initial_dir=os.path.join(root_dir, 'raw'))
        if not emg_file:
            print('No EMG file selected. Exiting.')
            return

    landmarks_file = args.landmarks_file
    if not landmarks_file:
        landmarks_file = prompt_file(title='Select landmarks (.npz) file', initial_dir=os.path.join(root_dir, 'media', 'landmarks'))
        if not landmarks_file:
            print('No landmarks file selected. Exiting.')
            return

    run_sync(root_dir, emg_file, landmarks_file, out_json=args.output, out_plot=args.plot, max_offset=args.max_offset)


if __name__ == '__main__':
    main()
