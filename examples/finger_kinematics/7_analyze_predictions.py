#!/usr/bin/env python3
"""
7_analyze_predictions.py - Comprehensive Prediction Analysis

STEP 7 (Optional): Analyze prediction quality with detailed diagnostics.

This script provides in-depth analysis of model predictions including:
- Per-joint error distributions and statistics
- Correlation analysis between predictions and ground truth
- Bland-Altman plots for agreement assessment
- Frequency domain analysis of prediction errors
- Cross-joint correlation analysis
- Time-series error evolution

Examples:
    # Analyze single prediction file
    python 7_analyze_predictions.py --pred_file predictions.txt --angles_file angles.csv
    
    # Batch analysis across multiple recordings
    python 7_analyze_predictions.py --root_dir /data --pattern "test_*.txt"
    
    # Generate comprehensive report
    python 7_analyze_predictions.py --pred_file preds.txt --angles_file angles.csv --save_report
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy import stats
from scipy.fft import fft, fftfreq

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def load_predictions(pred_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from text file."""
    data = []
    timestamps = []
    
    with open(pred_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamps.append(float(parts[0]))
                data.append([float(x) for x in parts[1:]])
    
    return np.array(timestamps), np.array(data)


def load_ground_truth(angles_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth angles from CSV."""
    df = pd.read_csv(angles_file)
    timestamps = df.iloc[:, 0].values
    angles = df.iloc[:, 1:].values
    return timestamps, angles


def interpolate_to_common_time(
    t1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, y2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both signals to common time base."""
    from scipy.interpolate import interp1d
    
    # Use the finer time grid
    t_common = t1 if len(t1) > len(t2) else t2
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    t_common = t_common[(t_common >= t_start) & (t_common <= t_end)]
    
    # Interpolate both to common time
    f1 = interp1d(t1, y1, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    f2 = interp1d(t2, y2, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    y1_interp = f1(t_common)
    y2_interp = f2(t_common)
    
    return t_common, y1_interp, y2_interp


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute comprehensive error metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    errors = y_pred - y_true
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'error_mean': np.mean(errors),
        'error_std': np.std(errors),
        'error_median': np.median(errors),
        'error_iqr': np.percentile(errors, 75) - np.percentile(errors, 25),
        'max_error': np.max(np.abs(errors)),
        'pearson_r': stats.pearsonr(y_true.flatten(), y_pred.flatten())[0],
    }
    
    return metrics


def plot_correlation_analysis(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    joint_names: Optional[list] = None,
    save_path: Optional[str] = None
):
    """Create correlation scatter plots for each joint."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'Joint {i}' for i in range(n_joints)]
    
    fig, axes = plt.subplots(1, n_joints, figsize=(4*n_joints, 4))
    if n_joints == 1:
        axes = [axes]
    
    for i in range(n_joints):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10)
        
        # Perfect prediction line
        lim_min = min(y_true[:, i].min(), y_pred[:, i].min())
        lim_max = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='Perfect', alpha=0.7)
        
        # Linear fit
        m, b = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        x_fit = np.array([lim_min, lim_max])
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, 'g-', label=f'Fit: y={m:.2f}x+{b:.1f}', alpha=0.7)
        
        # Compute R²
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        pearson_r, _ = stats.pearsonr(y_true[:, i], y_pred[:, i])
        
        ax.set_xlabel(f'True {joint_names[i]} (°)')
        ax.set_ylabel(f'Predicted {joint_names[i]} (°)')
        ax.set_title(f'{joint_names[i]}\nR²={r2:.3f}, r={pearson_r:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"   Saved correlation plot: {save_path}")
    plt.show()


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_names: Optional[list] = None,
    save_path: Optional[str] = None
):
    """Create Bland-Altman plots for agreement analysis."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'Joint {i}' for i in range(n_joints)]
    
    fig, axes = plt.subplots(1, n_joints, figsize=(4*n_joints, 4))
    if n_joints == 1:
        axes = [axes]
    
    for i in range(n_joints):
        ax = axes[i]
        
        mean = (y_true[:, i] + y_pred[:, i]) / 2
        diff = y_pred[:, i] - y_true[:, i]
        
        # Scatter plot
        ax.scatter(mean, diff, alpha=0.3, s=10)
        
        # Mean difference and limits of agreement
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        
        ax.axhline(mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.2f}°')
        ax.axhline(loa_upper, color='red', linestyle='--', label=f'+1.96σ: {loa_upper:.2f}°')
        ax.axhline(loa_lower, color='red', linestyle='--', label=f'-1.96σ: {loa_lower:.2f}°')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(f'Mean of True and Predicted (°)')
        ax.set_ylabel(f'Difference (Predicted - True) (°)')
        ax.set_title(f'{joint_names[i]}\nBland-Altman Plot')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"   Saved Bland-Altman plot: {save_path}")
    plt.show()


def plot_error_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_names: Optional[list] = None,
    save_path: Optional[str] = None
):
    """Plot error distributions with statistics."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'Joint {i}' for i in range(n_joints)]
    
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(2, n_joints, figsize=(4*n_joints, 8))
    if n_joints == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_joints):
        # Histogram
        ax = axes[0, i]
        ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='Zero error')
        ax.axvline(np.mean(errors[:, i]), color='blue', linestyle='-', 
                  label=f'Mean: {np.mean(errors[:, i]):.2f}°')
        ax.set_xlabel('Prediction Error (°)')
        ax.set_ylabel('Count')
        ax.set_title(f'{joint_names[i]}\nError Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[1, i]
        stats.probplot(errors[:, i], dist="norm", plot=ax)
        ax.set_title(f'{joint_names[i]}\nQ-Q Plot (Normality Check)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"   Saved error distribution plot: {save_path}")
    plt.show()


def plot_frequency_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    joint_names: Optional[list] = None,
    save_path: Optional[str] = None
):
    """Analyze frequency content of errors."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'Joint {i}' for i in range(n_joints)]
    
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(1, n_joints, figsize=(4*n_joints, 4))
    if n_joints == 1:
        axes = [axes]
    
    for i in range(n_joints):
        ax = axes[i]
        
        # Compute FFT
        N = len(errors[:, i])
        yf = fft(errors[:, i])
        xf = fftfreq(N, 1/fs)[:N//2]
        
        # Power spectrum
        power = 2.0/N * np.abs(yf[0:N//2])
        
        ax.plot(xf, power)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'{joint_names[i]}\nError Frequency Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(10, fs/2)])  # Focus on 0-10 Hz
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"   Saved frequency analysis plot: {save_path}")
    plt.show()


def plot_time_series_errors(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_names: Optional[list] = None,
    save_path: Optional[str] = None
):
    """Plot time-series of predictions with error shading."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'Joint {i}' for i in range(n_joints)]
    
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3*n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    
    for i in range(n_joints):
        ax = axes[i]
        
        # Plot predictions and ground truth
        ax.plot(timestamps, y_true[:, i], label='Ground Truth', alpha=0.7, linewidth=1.5)
        ax.plot(timestamps, y_pred[:, i], label='Predicted', alpha=0.7, linewidth=1.5)
        
        # Fill error regions
        ax.fill_between(timestamps, y_true[:, i], y_pred[:, i], 
                        alpha=0.3, label='Error', color='red')
        
        # Compute rolling MAE
        window = 50
        if len(errors) > window:
            rolling_mae = pd.Series(np.abs(errors[:, i])).rolling(window=window).mean()
            ax2 = ax.twinx()
            ax2.plot(timestamps, rolling_mae, 'r--', alpha=0.5, linewidth=1, 
                    label=f'Rolling MAE (w={window})')
            ax2.set_ylabel('Rolling MAE (°)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='upper right', fontsize=8)
        
        ax.set_ylabel(f'{joint_names[i]} (°)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        rmse = np.sqrt(np.mean(errors[:, i]**2))
        mae = np.mean(np.abs(errors[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        textstr = f'RMSE={rmse:.2f}°\nMAE={mae:.2f}°\nR²={r2:.3f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"   Saved time-series plot: {save_path}")
    plt.show()


def plot_cross_correlation_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_names: Optional[list] = None,
    save_path: Optional[str] = None
):
    """Plot correlation matrix between joints."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'J{i}' for i in range(n_joints)]
    
    # Compute correlation matrices
    corr_true = np.corrcoef(y_true.T)
    corr_pred = np.corrcoef(y_pred.T)
    corr_diff = corr_pred - corr_true
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth correlations
    im0 = axes[0].imshow(corr_true, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Ground Truth\nJoint Correlations')
    axes[0].set_xticks(range(n_joints))
    axes[0].set_yticks(range(n_joints))
    axes[0].set_xticklabels(joint_names)
    axes[0].set_yticklabels(joint_names)
    plt.colorbar(im0, ax=axes[0])
    
    # Predicted correlations
    im1 = axes[1].imshow(corr_pred, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Predicted\nJoint Correlations')
    axes[1].set_xticks(range(n_joints))
    axes[1].set_yticks(range(n_joints))
    axes[1].set_xticklabels(joint_names)
    axes[1].set_yticklabels(joint_names)
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    im2 = axes[2].imshow(corr_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference\n(Pred - True)')
    axes[2].set_xticks(range(n_joints))
    axes[2].set_yticks(range(n_joints))
    axes[2].set_xticklabels(joint_names)
    axes[2].set_yticklabels(joint_names)
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"   Saved correlation matrix: {save_path}")
    plt.show()


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_names: Optional[list] = None,
    output_dir: Optional[str] = None
):
    """Generate comprehensive analysis report."""
    n_joints = y_true.shape[1]
    joint_names = joint_names or [f'Joint {i}' for i in range(n_joints)]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info("\n=== PREDICTION ANALYSIS REPORT ===\n")
    
    # Overall metrics
    overall_metrics = compute_metrics(y_true, y_pred)
    logging.info("Overall Metrics:")
    for key, val in overall_metrics.items():
        logging.info(f"   {key.upper()}: {val:.4f}")
    
    # Per-joint metrics
    logging.info("\nPer-Joint Metrics:")
    for i in range(n_joints):
        metrics = compute_metrics(y_true[:, i:i+1], y_pred[:, i:i+1])
        logging.info(f"\n   {joint_names[i]}:")
        for key, val in metrics.items():
            logging.info(f"      {key.upper()}: {val:.4f}")
    
    # Generate plots
    if output_dir:
        plot_correlation_analysis(y_true, y_pred, joint_names, 
                                  os.path.join(output_dir, 'correlation.png'))
        plot_bland_altman(y_true, y_pred, joint_names,
                         os.path.join(output_dir, 'bland_altman.png'))
        plot_error_distributions(y_true, y_pred, joint_names,
                                os.path.join(output_dir, 'error_dist.png'))
        plot_cross_correlation_matrix(y_true, y_pred, joint_names,
                                      os.path.join(output_dir, 'cross_corr.png'))
    else:
        plot_correlation_analysis(y_true, y_pred, joint_names)
        plot_bland_altman(y_true, y_pred, joint_names)
        plot_error_distributions(y_true, y_pred, joint_names)
        plot_cross_correlation_matrix(y_true, y_pred, joint_names)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--pred_file", required=True, help="Prediction file (.txt)")
    parser.add_argument("--angles_file", required=True, help="Ground truth angles CSV")
    parser.add_argument("--output_dir", help="Output directory for plots")
    parser.add_argument("--fs", type=float, help="Sampling frequency for frequency analysis")
    parser.add_argument("--joint_names", nargs='+', help="Joint names (e.g., Thumb Index Middle Ring Pinky)")
    
    args = parser.parse_args()
    
    # Load data
    logging.info(f"Loading predictions: {args.pred_file}")
    pred_times, y_pred = load_predictions(args.pred_file)
    
    logging.info(f"Loading ground truth: {args.angles_file}")
    true_times, y_true = load_ground_truth(args.angles_file)
    
    # Interpolate to common time base
    logging.info("Aligning time series...")
    timestamps, y_true_aligned, y_pred_aligned = interpolate_to_common_time(
        true_times, y_true, pred_times, y_pred
    )
    
    logging.info(f"Analyzing {len(timestamps)} samples, {y_true_aligned.shape[1]} joints")
    
    # Generate report
    generate_report(y_true_aligned, y_pred_aligned, args.joint_names, args.output_dir)
    
    # Time-series and frequency plots (if requested)
    if args.output_dir:
        plot_time_series_errors(timestamps, y_true_aligned, y_pred_aligned, 
                               args.joint_names, 
                               os.path.join(args.output_dir, 'time_series.png'))
        
        if args.fs:
            plot_frequency_analysis(y_true_aligned, y_pred_aligned, args.fs,
                                   args.joint_names,
                                   os.path.join(args.output_dir, 'frequency.png'))
    
    logging.info(f"\n✅ Analysis complete!")
    if args.output_dir:
        logging.info(f"   Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
