#!/usr/bin/env python3
"""
1_synchronize_emg_video.py - EMG-Video Temporal Alignment

STEP 1: Synchronizes EMG recordings with video-based landmark data.

This script MUST be run BEFORE building datasets to ensure correct temporal
alignment between EMG features and joint angle labels.

Problem:
    EMG and video recordings may start/stop at different times, leading to
    misaligned features and labels. Offsets can be several seconds!

Solution:
    Cross-correlation of EMG envelope with landmark movement velocity to find
    the optimal time shift.

Usage:
    # Find offset between EMG and video (single pair)
    python 1_synchronize_emg_video.py find \\
        --root_dir /path/to/data \\
        --emg_file recording.rhd \\
        --landmarks_file recording_landmarks.npz
    
    # Batch process all recordings (RECOMMENDED)
    python 1_synchronize_emg_video.py batch \\
        --root_dir /path/to/data \\
        --verbose
    
    # Visualize alignment quality
    python 1_synchronize_emg_video.py plot \\
        --root_dir /path/to/data \\
        --emg_file recording.rhd \\
        --landmarks_file recording_landmarks.npz \\
        --offset_file recording_sync.json

Expected Directory Structure:
    root_dir/
        raw/                           # EMG recordings
            recording1/
                recording1.rhd
        media/landmarks/               # Landmark data
            recording1_landmarks.npz
        sync/                          # Output: Sync offsets (auto-created)
            recording1_sync.json

Output Format (JSON):
    {
        "offset_sec": 9.691,           # Time offset in seconds
        "offset_samples_emg": 9691,    # Offset in EMG samples
        "offset_samples_landmarks": 197,  # Offset in video frames
        "correlation_peak": 15739.39,  # Peak correlation value
        "confidence": 0.49,            # Confidence metric (0-1)
        "emg_fs": 1000.0,
        "landmark_fs": 20.4
    }

Interpretation:
    - Positive offset: Landmarks are DELAYED relative to EMG
    - To align: landmark_times_aligned = landmark_times - offset_sec
    - Confidence >0.3: Good alignment
    - Confidence <0.3: Manual verification recommended

Author: Neuro-Mechatronics Lab
Date: 2025-12-09
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, Dict
from glob import glob
import re

import numpy as np
import matplotlib.pyplot as plt

from intan.io import load_single_file
from intan.processing import (
    compute_landmark_movement_signal,
    compute_emg_envelope_signal,
    find_sync_offset,
    save_sync_offset
)

# Make the package importable when running examples from the repo root
try:
    import intan  # noqa: F401
except Exception:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import intan  # noqa: F401
    except Exception:
        print('\n[ERROR] Cannot import `intan`.')
        print('Either install the package in editable mode:')
        print('  pip install -e .')
        print('or run this script from the repository after installing dependencies.')
        sys.exit(1)


def synchronize_pair(
    root_dir: str,
    emg_file: str,
    landmarks_file: str,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Find synchronization offset for EMG-landmark pair.
    
    Args:
        root_dir: Project root directory
        emg_file: Path to EMG recording (.rhd)
        landmarks_file: Path to landmarks file (.npz)
        output_file: Path to save sync info (JSON)
        verbose: Verbose logging
        
    Returns:
        Synchronization info dict
    """
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    logging.info(f"[Synchronizing]")
    logging.info(f"   EMG: {os.path.basename(emg_file)}")
    logging.info(f"   Landmarks: {os.path.basename(landmarks_file)}")
    
    # Load EMG data
    data = load_single_file('rhd', emg_file, root_dir, verbose=False)
    emg_data = data['amplifier_data']
    emg_fs = data['frequency_parameters']['amplifier_sample_rate']
    
    logging.info(f"   EMG: {emg_data.shape[0]} channels, {emg_data.shape[1]} samples @ {emg_fs:.0f} Hz")
    logging.info(f"   Duration: {emg_data.shape[1] / emg_fs:.2f} seconds")
    
    # Load landmarks
    landmarks_data = np.load(landmarks_file)
    landmarks = landmarks_data['landmarks']
    landmark_times = landmarks_data['time_vector']
    landmark_fs = landmarks_data['sampling_rate']
    
    logging.info(f"   Landmarks: {landmarks.shape[0]} frames @ {landmark_fs:.1f} fps")
    logging.info(f"   Duration: {landmark_times[-1]:.2f} seconds")
    
    # Compute signals
    logging.info(f"   Computing movement signal from landmarks...")
    landmark_signal, _ = compute_landmark_movement_signal(landmarks, landmark_times, method='velocity')
    
    logging.info(f"   Computing EMG envelope signal...")
    emg_signal, emg_times = compute_emg_envelope_signal(emg_data, emg_fs)
    
    # Find offset
    logging.info(f"   Cross-correlating signals...")
    sync_info = find_sync_offset(emg_signal, emg_times, landmark_signal, landmark_times)
    
    logging.info(f"\n[Results]")
    logging.info(f"   Time offset: {sync_info['offset_sec']:.3f} seconds")
    logging.info(f"   EMG offset: {sync_info['offset_samples_emg']} samples")
    logging.info(f"   Landmark offset: {sync_info['offset_samples_landmarks']} frames")
    logging.info(f"   Confidence: {sync_info['confidence']:.2f}")
    
    if sync_info['confidence'] < 0.3:
        logging.warning(f"   [WARNING] Low confidence! Manual verification recommended.")
    
    # Add metadata
    sync_info['emg_file'] = emg_file
    sync_info['landmarks_file'] = landmarks_file
    
    # Save
    if output_file:
        save_sync_offset(sync_info, output_file)
        logging.info(f"\n[OK] Saved to: {output_file}")
    
    return sync_info


def batch_synchronize(
    root_dir: str,
    output_dir: Optional[str] = None,
    verbose: bool = False
):
    """
    Batch synchronize all EMG-landmark pairs in directory.
    
    Args:
        root_dir: Project root directory
        output_dir: Output directory for sync files (default: root_dir/sync)
        verbose: Verbose logging
    """
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    if output_dir is None:
        output_dir = os.path.join(root_dir, 'sync')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"[Batch Synchronization]")
    logging.info(f"   Root: {root_dir}")
    logging.info(f"   Output: {output_dir}")
    
    # Find all RHD files
    raw_dir = os.path.join(root_dir, 'raw')
    landmarks_dir = os.path.join(root_dir, 'media', 'landmarks')
    
    # Get list of recordings
    emg_files = glob(os.path.join(raw_dir, '**', '*.rhd'), recursive=True)
    
    logging.info(f"\n[Found {len(emg_files)} EMG recordings]")
    
    results = []
    for emg_file in emg_files:
        # Get base name
        base_name = os.path.basename(emg_file).replace('.rhd', '')
        
        # Try to find matching landmarks
        landmark_patterns = [
            os.path.join(landmarks_dir, f"{base_name}_landmarks.npz"),
            os.path.join(landmarks_dir, f"{base_name}.npz"),
        ]
        
        # Also try without timestamp suffix
        base_no_timestamp = re.sub(r'_\d{6}_\d{6}$', '', base_name)
        if base_no_timestamp != base_name:
            landmark_patterns.extend([
                os.path.join(landmarks_dir, f"{base_no_timestamp}_landmarks.npz"),
                os.path.join(landmarks_dir, f"{base_no_timestamp}.npz"),
            ])
        
        landmarks_file = None
        for pattern in landmark_patterns:
            if os.path.exists(pattern):
                landmarks_file = pattern
                break
        
        if not landmarks_file:
            logging.warning(f"   [SKIP] No landmarks found for: {base_name}")
            continue
        
        # Output file
        output_file = os.path.join(output_dir, f"{base_name}_sync.json")
        
        logging.info(f"\n[{len(results)+1}] {base_name}")
        
        try:
            sync_info = synchronize_pair(
                root_dir=root_dir,
                emg_file=emg_file,
                landmarks_file=landmarks_file,
                output_file=output_file,
                verbose=False
            )
            results.append(sync_info)
            
        except Exception as e:
            logging.error(f"   [FAIL] {e}")
    
    # Summary
    if results:
        offsets = [r['offset_sec'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        logging.info(f"\n[Summary - {len(results)} synchronized]")
        logging.info(f"   Mean offset: {np.mean(offsets):.3f} ± {np.std(offsets):.3f} sec")
        logging.info(f"   Mean confidence: {np.mean(confidences):.2f}")
        logging.info(f"   Low confidence (<0.3): {sum(c < 0.3 for c in confidences)} files")
        logging.info(f"\n[NEXT STEP] Run 2_extract_joint_angles.py (if not done)")
        logging.info(f"            Then run 3_build_dataset.py to create training dataset")


def plot_synchronization(
    root_dir: str,
    emg_file: str,
    landmarks_file: str,
    offset_file: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize synchronization between EMG and landmarks.
    
    Args:
        root_dir: Project root
        emg_file: EMG file path
        landmarks_file: Landmarks file path
        offset_file: Optional sync offset JSON
        save_path: Optional path to save figure
    """
    # Load data
    data = load_single_file('rhd', emg_file, root_dir, verbose=False)
    emg_data = data['amplifier_data']
    emg_fs = data['frequency_parameters']['amplifier_sample_rate']
    
    landmarks_data = np.load(landmarks_file)
    landmarks = landmarks_data['landmarks']
    landmark_times = landmarks_data['time_vector']
    
    # Compute signals
    from intan.processing import load_sync_offset as load_offset_from_file
    landmark_signal, _ = compute_landmark_movement_signal(landmarks, landmark_times)
    emg_signal, emg_times = compute_emg_envelope_signal(emg_data, emg_fs)
    
    # Load offset if provided
    offset_sec = 0.0
    if offset_file and os.path.exists(offset_file):
        offset_sec = load_offset_from_file(offset_file)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # EMG envelope
    axes[0].plot(emg_times, emg_signal, label='EMG Envelope', alpha=0.7)
    axes[0].set_ylabel('EMG Envelope (normalized)')
    axes[0].set_title('EMG Signal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Landmark movement (aligned)
    landmark_times_aligned = landmark_times - offset_sec
    axes[1].plot(landmark_times_aligned, landmark_signal, 
                 label=f'Landmark Movement (offset applied: {offset_sec:.3f}s)', 
                 color='orange', alpha=0.7)
    axes[1].set_ylabel('Movement Velocity (normalized)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_title('Landmark Movement Signal (Aligned)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="STEP 1: Synchronize EMG and Video Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Batch synchronize all recordings (RECOMMENDED first step)
    python 1_synchronize_emg_video.py batch --root_dir /data --verbose
    
    # Single file
    python 1_synchronize_emg_video.py find \\
        --root_dir /data \\
        --emg_file raw/recording.rhd \\
        --landmarks_file media/landmarks/recording_landmarks.npz
    
    # Visualize alignment
    python 1_synchronize_emg_video.py plot \\
        --root_dir /data \\
        --emg_file raw/recording.rhd \\
        --landmarks_file media/landmarks/recording_landmarks.npz \\
        --offset_file sync/recording_sync.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Find command
    find_parser = subparsers.add_parser('find', help='Find sync offset for single pair')
    find_parser.add_argument('--root_dir', required=True, help='Project root directory')
    find_parser.add_argument('--emg_file', required=True, help='EMG recording file (relative to root)')
    find_parser.add_argument('--landmarks_file', required=True, help='Landmarks NPZ file (relative to root)')
    find_parser.add_argument('--output', help='Output JSON file for sync info')
    find_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch synchronize all files (RECOMMENDED)')
    batch_parser.add_argument('--root_dir', required=True, help='Project root directory')
    batch_parser.add_argument('--output_dir', help='Output directory for sync files (default: root_dir/sync)')
    batch_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Visualize synchronization quality')
    plot_parser.add_argument('--root_dir', required=True, help='Project root directory')
    plot_parser.add_argument('--emg_file', required=True, help='EMG recording file')
    plot_parser.add_argument('--landmarks_file', required=True, help='Landmarks NPZ file')
    plot_parser.add_argument('--offset_file', help='Sync offset JSON file')
    plot_parser.add_argument('--save', help='Save figure to file instead of showing')
    
    args = parser.parse_args()
    
    if args.command == 'find':
        synchronize_pair(
            root_dir=args.root_dir,
            emg_file=args.emg_file,
            landmarks_file=args.landmarks_file,
            output_file=args.output,
            verbose=args.verbose
        )
    
    elif args.command == 'batch':
        batch_synchronize(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    elif args.command == 'plot':
        plot_synchronization(
            root_dir=args.root_dir,
            emg_file=args.emg_file,
            landmarks_file=args.landmarks_file,
            offset_file=args.offset_file,
            save_path=args.save
        )
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
