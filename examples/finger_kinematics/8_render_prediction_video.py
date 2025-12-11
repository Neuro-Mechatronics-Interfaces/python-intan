#!/usr/bin/env python3
"""
8_render_prediction_video.py - Video Overlay Tool for Hand Predictions

STEP 8 (Optional): Render annotated video with prediction overlay.

This script creates visualization videos with:
- Hand skeleton overlay with joint angle annotations
- Side-by-side comparison of ground truth vs predicted angles
- Color-coded error visualization (green=good, red=high error)
- Real-time angle plots at bottom of frame
- Synchronized video playback with predictions

Examples:
    # Basic overlay
    python 8_render_prediction_video.py --video test.mp4 --pred predictions.txt --landmarks landmarks.npz
    
    # Side-by-side comparison
    python 8_render_prediction_video.py --video test.mp4 --pred predictions.txt \
                                        --angles angles.csv --landmarks landmarks.npz \
                                        --layout sidebyside
    
    # Error heatmap overlay
    python 8_render_prediction_video.py --video test.mp4 --pred predictions.txt \
                                        --angles angles.csv --landmarks landmarks.npz \
                                        --show_errors --error_threshold 15
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from scipy.interpolate import interp1d

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm connections
]

# Joint indices for finger angles (wrist, mcp, pip, dip, tip)
FINGER_JOINTS = {
    'Thumb': [0, 1, 2, 3, 4],
    'Index': [0, 5, 6, 7, 8],
    'Middle': [0, 9, 10, 11, 12],
    'Ring': [0, 13, 14, 15, 16],
    'Pinky': [0, 17, 18, 19, 20]
}


def load_predictions(pred_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from text file (CSV or space-separated)."""
    data = []
    timestamps = []
    
    with open(pred_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            # Support both comma and space separation
            if ',' in line:
                parts = line.strip().split(',')
            else:
                parts = line.strip().split()
            
            if len(parts) >= 2:
                timestamps.append(float(parts[0]))
                data.append([float(x) for x in parts[1:]])
    
    return np.array(timestamps), np.array(data)


def load_landmarks(landmarks_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load hand landmarks from npz."""
    data = np.load(landmarks_file)
    return data['time_vector'], data['landmarks']


def load_ground_truth(angles_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth angles from CSV."""
    df = pd.read_csv(angles_file)
    timestamps = df.iloc[:, 0].values
    angles = df.iloc[:, 1:].values
    return timestamps, angles


def interpolate_to_video(video_fps: float, video_frames: int, 
                        timestamps: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Interpolate data to match video frame rate."""
    video_times = np.arange(video_frames) / video_fps
    
    # Find overlap region
    t_start = max(video_times[0], timestamps[0])
    t_end = min(video_times[-1], timestamps[-1])
    
    # Interpolate
    f = interp1d(timestamps, data, axis=0, kind='linear', 
                bounds_error=False, fill_value='extrapolate')
    
    data_interp = f(video_times)
    
    return data_interp


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    connections: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
):
    """Draw hand skeleton on frame."""
    h, w = frame.shape[:2]
    
    # Draw connections
    for conn in connections:
        pt1 = landmarks[conn[0]]
        pt2 = landmarks[conn[1]]
        
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue
        
        x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
        x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
        
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw landmarks
    for pt in landmarks:
        if np.isnan(pt).any():
            continue
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(frame, (x, y), 4, color, -1)


def get_error_color(error: float, threshold: float = 15.0) -> Tuple[int, int, int]:
    """Get color based on error magnitude (BGR)."""
    # Handle NaN or invalid errors
    if np.isnan(error) or error < 0:
        return (128, 128, 128)  # Gray for invalid
    
    # Green (good) to Red (bad)
    if error < threshold / 2:
        return (0, 255, 0)  # Green
    elif error < threshold:
        # Interpolate green to yellow
        ratio = (error - threshold/2) / (threshold/2)
        g = 255
        r = int(255 * ratio)
        return (0, g, r)
    else:
        # Interpolate yellow to red
        ratio = min((error - threshold) / threshold, 1.0)
        g = int(255 * (1 - ratio))
        return (0, g, 255)


def draw_angle_annotation(
    frame: np.ndarray,
    landmarks: np.ndarray,
    finger_name: str,
    angle: float,
    error: Optional[float] = None,
    error_threshold: float = 15.0
):
    """Draw angle annotation near finger tip."""
    h, w = frame.shape[:2]
    
    # Get tip landmark
    tip_idx = FINGER_JOINTS[finger_name][-1]
    tip = landmarks[tip_idx]
    
    if np.isnan(tip).any():
        return
    
    x, y = int(tip[0] * w), int(tip[1] * h)
    
    # Choose color based on error
    if error is not None:
        color = get_error_color(error, error_threshold)
        text = f"{finger_name}: {angle:.1f}° (err:{error:.1f}°)"
    else:
        color = (0, 255, 0)
        text = f"{finger_name}: {angle:.1f}°"
    
    # Draw text with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text above finger tip
    text_x = x - text_w // 2
    text_y = y - 10
    
    # Ensure text stays in frame
    text_x = max(5, min(text_x, w - text_w - 5))
    text_y = max(text_h + 5, text_y)
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                 (text_x - 2, text_y - text_h - 2),
                 (text_x + text_w + 2, text_y + baseline + 2),
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


def create_angle_plot(
    timestamps: np.ndarray,
    angles_true: Optional[np.ndarray],
    angles_pred: np.ndarray,
    current_frame: int,
    fps: float,
    joint_names: List[str],
    window_sec: float = 5.0
) -> np.ndarray:
    """Create matplotlib plot of angles over time."""
    n_joints = angles_pred.shape[1]
    
    fig, axes = plt.subplots(n_joints, 1, figsize=(8, 2*n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    
    current_time = current_frame / fps
    t_start = max(0, current_time - window_sec)
    t_end = current_time + window_sec/5  # Show more history than future
    
    for i in range(n_joints):
        ax = axes[i]
        
        # Plot ground truth if available
        if angles_true is not None:
            ax.plot(timestamps, angles_true[:, i], 'b-', label='Ground Truth', alpha=0.7)
        
        # Plot predictions
        ax.plot(timestamps, angles_pred[:, i], 'r-', label='Predicted', alpha=0.7)
        
        # Mark current frame
        if current_frame < len(timestamps):
            ax.axvline(current_time, color='green', linestyle='--', linewidth=2)
        
        ax.set_xlim([t_start, t_end])
        ax.set_ylim([0, 180])
        ax.set_ylabel(f'{joint_names[i]} (°)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    
    plt.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib
    buf = np.asarray(fig.canvas.buffer_rgba())
    plot_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return plot_img


def render_video(
    video_path: str,
    pred_times: np.ndarray,
    pred_angles: np.ndarray,
    landmark_times: np.ndarray,
    landmarks: np.ndarray,
    output_path: str,
    true_times: Optional[np.ndarray] = None,
    true_angles: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None,
    layout: str = 'overlay',
    show_errors: bool = False,
    error_threshold: float = 15.0,
    show_plots: bool = True
):
    """Render annotated video."""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"Video: {width}x{height} @ {fps:.2f} fps, {frame_count} frames")
    
    # Interpolate all data to video frame rate
    logging.info("Interpolating data to video frame rate...")
    pred_angles_interp = interpolate_to_video(fps, frame_count, pred_times, pred_angles)
    landmarks_interp = interpolate_to_video(fps, frame_count, landmark_times, landmarks)
    
    if true_angles is not None:
        true_angles_interp = interpolate_to_video(fps, frame_count, true_times, true_angles)
        errors = np.abs(pred_angles_interp - true_angles_interp)
    else:
        true_angles_interp = None
        errors = None
    
    # Setup video writer
    if show_plots:
        output_height = height + 400  # Extra space for plots
    else:
        output_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
    
    joint_names = joint_names or [f'Joint {i}' for i in range(pred_angles.shape[1])]
    video_times = np.arange(frame_count) / fps
    
    # Process frames
    logging.info("Rendering frames...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw skeleton
        if frame_idx < len(landmarks_interp):
            lm = landmarks_interp[frame_idx]
            if not np.isnan(lm).all():
                draw_skeleton(frame, lm, HAND_CONNECTIONS, color=(0, 255, 0), thickness=2)
                
                # Draw angle annotations
                for i, (finger_name, angle) in enumerate(zip(joint_names, pred_angles_interp[frame_idx])):
                    error = errors[frame_idx, i] if errors is not None else None
                    if show_errors and error is not None:
                        draw_angle_annotation(frame, lm, finger_name, angle, error, error_threshold)
                    else:
                        draw_angle_annotation(frame, lm, finger_name, angle)
        
        # Add frame number and time
        time_text = f"Frame: {frame_idx}/{frame_count} | Time: {video_times[frame_idx]:.2f}s"
        cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Add error stats if available
        if errors is not None and frame_idx < len(errors):
            mean_error = np.mean(errors[frame_idx])
            max_error = np.max(errors[frame_idx])
            error_text = f"Error: Mean={mean_error:.1f}° Max={max_error:.1f}°"
            cv2.putText(frame, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
        
        # Add angle plot at bottom
        if show_plots:
            plot_img = create_angle_plot(
                video_times[:frame_idx+1],
                true_angles_interp[:frame_idx+1] if true_angles_interp is not None else None,
                pred_angles_interp[:frame_idx+1],
                frame_idx, fps, joint_names, window_sec=5.0
            )
            
            # Resize plot to match video width
            plot_img = cv2.resize(plot_img, (width, 400))
            
            # Combine video frame and plot
            combined = np.vstack([frame, plot_img])
        else:
            combined = frame
        
        out.write(combined)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            logging.info(f"   Processed {frame_idx}/{frame_count} frames ({100*frame_idx/frame_count:.1f}%)")
    
    cap.release()
    out.release()
    
    logging.info(f"✅ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--pred", required=True, help="Prediction file (.txt)")
    parser.add_argument("--landmarks", required=True, help="Landmarks file (.npz)")
    parser.add_argument("--angles", help="Ground truth angles CSV (optional)")
    parser.add_argument("--output", help="Output video path (default: input_annotated.mp4)")
    parser.add_argument("--joint_names", nargs='+', 
                       default=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'],
                       help="Joint names")
    parser.add_argument("--layout", choices=['overlay', 'sidebyside'], default='overlay',
                       help="Video layout")
    parser.add_argument("--show_errors", action="store_true",
                       help="Show error values (requires --angles)")
    parser.add_argument("--error_threshold", type=float, default=15.0,
                       help="Error threshold for color coding (degrees)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Disable time-series plots at bottom")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_stem = Path(args.video).stem
        output_path = str(Path(args.video).parent / f"{video_stem}_annotated.mp4")
    
    # Load data
    logging.info(f"Loading predictions: {args.pred}")
    pred_times, pred_angles = load_predictions(args.pred)
    
    logging.info(f"Loading landmarks: {args.landmarks}")
    landmark_times, landmarks = load_landmarks(args.landmarks)
    
    true_times, true_angles = None, None
    if args.angles:
        logging.info(f"Loading ground truth: {args.angles}")
        true_times, true_angles = load_ground_truth(args.angles)
    
    # Render video
    render_video(
        video_path=args.video,
        pred_times=pred_times,
        pred_angles=pred_angles,
        landmark_times=landmark_times,
        landmarks=landmarks,
        output_path=output_path,
        true_times=true_times,
        true_angles=true_angles,
        joint_names=args.joint_names,
        layout=args.layout,
        show_errors=args.show_errors,
        error_threshold=args.error_threshold,
        show_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
