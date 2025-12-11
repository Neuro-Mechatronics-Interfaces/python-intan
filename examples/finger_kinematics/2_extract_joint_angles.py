#!/usr/bin/env python3
"""
2_extract_joint_angles.py - Extract Joint Angles from Hand Landmark Video

STEP 2: Extract joint angles from video landmarks.

Prerequisite: Run 1_synchronize_emg_video.py first to generate sync offsets!

This script processes a video file to extract hand landmarks using MediaPipe,
then converts the 21 3D landmarks into joint angles for each finger.

Pipeline:
1. Extract hand landmarks from video (21 points × XYZ coordinates)
2. Apply Kalman filtering for smooth trajectories
3. Convert landmarks to joint angles (5 angles: thumb, index, middle, ring, pinky)
4. Save as CSV file with timestamps for EMG-to-kinematics training

Output CSV format:
    timestamp,thumb,index,middle,ring,pinky
    0.000,145.2,160.8,165.3,158.7,152.4
    0.033,145.5,161.2,165.8,159.1,152.9
    ...

Joint Angle Definition:
    Each angle is computed at the MCP (metacarpophalangeal) joint:
    angle = fingertip → MCP → wrist
    
Examples:
    # Interactive mode - file dialog will open
    python 1_extract_joint_angles.py
    
    # Specify video path directly
    python 1_extract_joint_angles.py --video_path /path/to/video.mp4
    
    # With live visualization during processing
    python 1_extract_joint_angles.py --video_path video.mp4 --visualize --save_video
    
    # Custom output directory for landmarks/
    python 1_extract_joint_angles.py --video_path video.mp4 --output_dir /path/to/output/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import python-intan file dialog utilities
from intan.io import prompt_file


def safe_relpath(path: str, start: str = None) -> str:
    """
    Get relative path, falling back to absolute path if on different drives (Windows).
    
    Args:
        path: Target path
        start: Starting path (defaults to current directory)
        
    Returns:
        Relative path if possible, otherwise absolute path
    """
    try:
        return os.path.relpath(path, start)
    except (ValueError, TypeError):
        # ValueError: different drives on Windows
        # TypeError: None path
        return os.path.abspath(path)


# Check if handtrack package is available
try:
    from handtrack.tracker import HandTracker
    from handtrack.processing import compute_finger_angles
except ImportError as e:
    print("\n[ERROR] Hand-Landmark-Tracker package not found.")
    print("\nThis script requires the 'handtrack' package and its dependencies.")
    print("\nInstallation options:")
    print("  1. Install video extras (recommended):")
    print("     pip install 'python-intan[video]'")
    print("\n  2. Install manually:")
    print("     pip install opencv-python mediapipe")
    print("     pip install git+https://github.com/Jshulgach/Hand-Landmark-Tracker.git")
    print("\n  3. Install from local clone:")
    print("     cd _review/Hand-Landmark-Tracker")
    print("     pip install -e .")
    print(f"\nOriginal error: {e}\n")
    sys.exit(1)


def extract_landmarks_from_video(video_path: str, visualize: bool = False, save_video: bool = False) -> tuple:
    """
    Extract hand landmarks from video using MediaPipe + Kalman filtering.
    
    Args:
        video_path: Path to video file
        visualize: Show live tracking visualization
        save_video: Save annotated video
        
    Returns:
        landmarks: (n_frames, 21, 3) array
        sampling_rate: Video frame rate
        landmark_labels: List of landmark names
    """
    logging.info(f"[Step 1/3] Extracting landmarks from video...")
    logging.info(f"   Video: {os.path.basename(video_path)}")
    
    tracker = HandTracker(source=video_path, apply_kalman=True)
    landmarks, metadata = tracker.extract_landmarks(visualize=visualize, save_video=save_video)
    
    logging.info(f"   [OK] Extracted {landmarks.shape[0]} frames at {metadata['sampling_rate']:.1f} fps")
    logging.info(f"   [OK] Landmarks shape: {landmarks.shape} (frames, joints, xyz)")
    
    return landmarks, metadata['sampling_rate'], metadata['landmark_labels']


def compute_joint_angles_from_landmarks(landmarks: np.ndarray, sampling_rate: float) -> tuple:
    """
    Convert landmarks to joint angles for each frame.
    
    Args:
        landmarks: (n_frames, 21, 3) array of 3D hand landmarks
        sampling_rate: Video frame rate (fps)
        
    Returns:
        timestamps: (n_frames,) array of time values
        angles_df: DataFrame with columns [timestamp, thumb, index, middle, ring, pinky]
    """
    logging.info(f"[Step 2/3] Computing joint angles from landmarks...")
    
    n_frames = landmarks.shape[0]
    timestamps = np.arange(n_frames) / sampling_rate
    
    angles_list = []
    for i, frame_landmarks in enumerate(landmarks):
        angle_dict = compute_finger_angles(frame_landmarks)
        angles_list.append([
            angle_dict['thumb'],
            angle_dict['index'],
            angle_dict['middle'],
            angle_dict['ring'],
            angle_dict['pinky']
        ])
        
        if (i + 1) % 500 == 0 or i == n_frames - 1:
            logging.info(f"   Processing frames... {i+1}/{n_frames}")
    
    # Create DataFrame
    angles_array = np.array(angles_list)
    angles_df = pd.DataFrame(
        angles_array,
        columns=['thumb', 'index', 'middle', 'ring', 'pinky']
    )
    angles_df.insert(0, 'timestamp', timestamps)
    
    logging.info(f"   [OK] Computed {n_frames} angle samples")
    logging.info(f"   [OK] Joint angles shape: {angles_array.shape} (frames, fingers)")
    
    return timestamps, angles_df


def save_outputs(video_path: str, landmarks: np.ndarray, landmark_labels: list, 
                sampling_rate: float, angles_df: pd.DataFrame, output_dir: str = None):
    """
    Save landmarks (NPZ) and joint angles (CSV) to disk.
    
    Args:
        video_path: Original video path
        landmarks: (n_frames, 21, 3) landmark array
        landmark_labels: Names of 21 landmarks
        sampling_rate: Frame rate
        angles_df: DataFrame with joint angles
        output_dir: Output directory (defaults to landmarks/ subfolder)
    """
    logging.info(f"[Step 3/3] Saving output files...")
    
    # Determine output directory
    if output_dir is None:
        video_dir = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(video_dir, "landmarks")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    landmarks_path = os.path.join(output_dir, f"{video_stem}_landmarks.npz")
    angles_path = os.path.join(output_dir, f"{video_stem}_angles.csv")
    
    # Save landmarks as NPZ
    np.savez(
        landmarks_path,
        landmarks=landmarks,
        landmark_labels=landmark_labels,
        sampling_rate=sampling_rate,
        time_vector=angles_df['timestamp'].values
    )
    logging.info(f"   Saved landmarks: {safe_relpath(landmarks_path)}")
    
    # Save joint angles as CSV
    angles_df.to_csv(angles_path, index=False, float_format='%.6f')
    logging.info(f"   Saved angles:    {safe_relpath(angles_path)}")
    
    # Print summary statistics
    logging.info(f"\n[Summary]")
    logging.info(f"   Duration: {angles_df['timestamp'].iloc[-1]:.2f} seconds")
    logging.info(f"   Frames: {len(angles_df)}")
    logging.info(f"   Frame rate: {sampling_rate:.2f} fps")
    logging.info(f"   Joint angle ranges (degrees):")
    for col in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        min_val = angles_df[col].min()
        max_val = angles_df[col].max()
        mean_val = angles_df[col].mean()
        logging.info(f"      {col.capitalize():8s}: {min_val:6.1f}° to {max_val:6.1f}°  (mean: {mean_val:6.1f}°)")
    
    return landmarks_path, angles_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks and compute joint angles from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode - file dialog opens for video selection
    python 1_extract_joint_angles.py
    
    # Direct path specification (skips dialog)
    python 1_extract_joint_angles.py --video_path recording.mp4
    
    # With live visualization of landmark tracking
    python 1_extract_joint_angles.py --video_path recording.mp4 --visualize
    
    # Custom output directory for landmarks/ folder
    python 1_extract_joint_angles.py --video_path recording.mp4 --output_dir /data/landmarks/
        """
    )
    
    parser.add_argument("--video_path", type=str, default="", 
                       help="Path to video file (opens file dialog if not provided)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for landmarks/ folder (defaults to video directory)")
    parser.add_argument("--visualize", action="store_true",
                       help="Show live tracking visualization")
    parser.add_argument("--save_video", action="store_true",
                       help="Save annotated video with landmarks drawn")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s'
    )
    
    # Get video path
    logging.info("[Finger Kinematics - Joint Angle Extraction]\n")
    
    video_path = args.video_path.strip() if args.video_path else None
    
    # Use file dialog if no path provided
    if not video_path:
        logging.info("Please select a video file...")
        video_path = prompt_file(
            title="Select Video File for Hand Landmark Tracking",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
    
    if not video_path:
        logging.error("[ERROR] No video file selected. Exiting.")
        sys.exit(1)
    
    if not os.path.exists(video_path):
        logging.error(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        # Step 1: Extract landmarks
        landmarks, sampling_rate, landmark_labels = extract_landmarks_from_video(
            video_path, 
            visualize=args.visualize, 
            save_video=args.save_video
        )
        
        # Step 2: Compute joint angles
        timestamps, angles_df = compute_joint_angles_from_landmarks(landmarks, sampling_rate)
        
        # Step 3: Save outputs
        landmarks_path, angles_path = save_outputs(
            video_path, 
            landmarks, 
            landmark_labels, 
            sampling_rate, 
            angles_df,
            output_dir=args.output_dir
        )
        
        logging.info(f"\n[OK] Processing complete!")
        logging.info(f"     Use the angles CSV file with 2_build_dataset.py:")
        logging.info(f"     python 2_build_dataset.py --angles_file {safe_relpath(angles_path)}")
        
    except KeyboardInterrupt:
        logging.info("\n[CANCELLED] Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n[ERROR] Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
