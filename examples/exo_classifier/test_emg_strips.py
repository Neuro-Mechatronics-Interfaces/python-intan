#!/usr/bin/env python3
"""
3_test_emg_strips.py

Test classification performance across different EMG electrode strips.
Tests each of the 10 horizontal strips to find which locations give best accuracy.

The 128-channel array is organized in 10 strips of ~13 channels each:
- Strip 1 (top): [13, 26, 39, 52, 64, 76, 89, 102, 115, 128]
- Strip 2: [12, 25, 38, 51, 63, 75, 88, 101, 114, 127]
- ...
- Strip 10 (bottom): [1, 14, 27, 40, 77, 90, 103, 116]

Usage:
------
python 3_test_emg_strips.py \
    --root_dir /path/to/data \
    --label exo_gestures \
    --output_dir ./strip_analysis \
    --verbose
"""

import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib

# Define the 10 EMG strips (1-indexed channels)
EMG_STRIPS = {
    1: [13, 26, 39, 52, 64, 76, 89, 102, 115, 128],
    2: [12, 25, 38, 51, 63, 75, 88, 101, 114, 127],
    3: [11, 24, 37, 50, 62, 74, 87, 100, 113, 126],
    4: [10, 23, 36, 49, 61, 73, 86, 99, 112, 125],
    5: [9, 22, 35, 48, 60, 72, 85, 98, 111, 124],
    6: [8, 21, 34, 47, 59, 71, 84, 97, 110, 123],
    7: [7, 20, 33, 46, 58, 70, 83, 96, 109, 122],
    8: [6, 19, 32, 45, 57, 69, 82, 95, 108, 121],
    9: [5, 18, 31, 44, 56, 68, 81, 94, 107, 120],
    10: [4, 17, 30, 43, 55, 67, 80, 93, 106, 119],
    # Note: Channels 1, 2, 3, 14, 15, 16, 27, 28, 29, 40, 41, 42, 53, 54, 65, 66,
    #       77, 78, 90, 91, 103, 104, 116, 117 are in partial rows at edges
}


class CompactEMGClassifier(nn.Module):
    """Smaller network for 10-channel subset testing."""

    def __init__(self, input_dim=10, output_dim=10):
        super(CompactEMGClassifier, self).__init__()

        # Smaller architecture for fewer channels
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, output_dim)
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


def train_and_evaluate_strip(
        X: np.ndarray,
        y: np.ndarray,
        strip_channels: List[int],
        n_classes: int,
        device: torch.device,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        n_folds: int = 5,
        random_seed: int = 42,
) -> Dict:
    """
    Train and evaluate a model on a specific channel subset using k-fold CV.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix (n_samples, 128)
    y : np.ndarray
        Labels (n_samples,)
    strip_channels : List[int]
        1-indexed channel numbers to use
    n_classes : int
        Number of gesture classes
    device : torch.device
        Training device
    epochs : int
        Training epochs per fold
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    n_folds : int
        Number of CV folds
    random_seed : int
        Random seed

    Returns
    -------
    Dict with keys: mean_accuracy, std_accuracy, fold_accuracies
    """

    # Convert to 0-indexed
    strip_indices = [ch - 1 for ch in strip_channels]

    # Select only these channels
    X_subset = X[:, strip_indices]

    # No PCA - use raw RMS features from selected channels
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # K-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_val = X_scaled[val_idx]
        y_val = y[val_idx]

        # Initialize model
        model = CompactEMGClassifier(
            input_dim=len(strip_channels),
            output_dim=n_classes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)

        # Training loop
        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            n_samples = X_train.shape[0]
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        fold_accuracies.append(best_val_acc)

    return {
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies, ddof=1),
        'fold_accuracies': fold_accuracies,
    }


def test_all_strips(
        root_dir: str,
        label: str,
        output_dir: str,
        dataset_path: str = None,
        epochs: int = 50,
        batch_size: int = 16,
        n_folds: int = 5,
        random_seed: int = 42,
        verbose: bool = False,
):
    """
    Test classification performance for each EMG strip.
    """

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    logging.info("=" * 70)
    logging.info("EMG STRIP COMPARISON")
    logging.info("=" * 70)
    logging.info(f"Testing {len(EMG_STRIPS)} electrode strips")
    logging.info(f"Each strip: ~10 channels")
    logging.info(f"Evaluation: {n_folds}-fold cross-validation")
    logging.info(f"Training: {epochs} epochs per fold")
    logging.info("=" * 70)

    # Setup paths
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if dataset_path is None:
        dataset_path = os.path.join(root_dir, f"{label}_training_dataset.npz")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    logging.info(f"\nLoading dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    X = data["X"]  # (n_samples, 128)
    y = data["y"]
    class_names = data["class_names"].tolist()

    logging.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} channels")
    logging.info(f"Classes: {class_names}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}\n")

    # Test each strip
    results = {}

    for strip_num, channels in EMG_STRIPS.items():
        logging.info(f"Testing Strip {strip_num}...")
        logging.info(f"  Channels: {channels}")

        result = train_and_evaluate_strip(
            X, y_encoded, channels, n_classes,
            device, epochs, batch_size, n_folds=n_folds, random_seed=random_seed
        )

        results[strip_num] = result

        logging.info(f"  Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        logging.info(f"  Fold accuracies: {[f'{a:.4f}' for a in result['fold_accuracies']]}\n")

    # Find best strip
    best_strip = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
    logging.info("=" * 70)
    logging.info(f"BEST STRIP: Strip {best_strip[0]}")
    logging.info(f"  Channels: {EMG_STRIPS[best_strip[0]]}")
    logging.info(f"  Accuracy: {best_strip[1]['mean_accuracy']:.4f} ± {best_strip[1]['std_accuracy']:.4f}")
    logging.info("=" * 70)

    # Save results
    results_path = os.path.join(output_dir, f"{label}_strip_comparison.json")
    results_json = {
        "strips": {
            str(k): {
                "channels": EMG_STRIPS[k],
                "mean_accuracy": float(v['mean_accuracy']),
                "std_accuracy": float(v['std_accuracy']),
                "fold_accuracies": [float(x) for x in v['fold_accuracies']],
            }
            for k, v in results.items()
        },
        "best_strip": {
            "strip_number": int(best_strip[0]),
            "channels": EMG_STRIPS[best_strip[0]],
            "mean_accuracy": float(best_strip[1]['mean_accuracy']),
            "std_accuracy": float(best_strip[1]['std_accuracy']),
        },
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "n_folds": n_folds,
            "random_seed": random_seed,
        }
    }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logging.info(f"\n✓ Results saved: {results_path}")

    # Create visualizations
    create_plots(results, output_dir, label)

    return results


def create_plots(results: Dict, output_dir: str, label: str):
    """Create visualization plots for strip comparison."""

    logging.info("\nCreating plots...")

    # Extract data
    strip_numbers = sorted(results.keys())
    mean_accs = [results[s]['mean_accuracy'] for s in strip_numbers]
    std_accs = [results[s]['std_accuracy'] for s in strip_numbers]

    # Set style
    sns.set_style("whitegrid")

    # ========================================================================
    # Plot 1: Bar chart with error bars
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(strip_numbers, mean_accs, yerr=std_accs,
                  capsize=5, alpha=0.8, color='steelblue', edgecolor='black')

    # Highlight best strip
    best_idx = np.argmax(mean_accs)
    bars[best_idx].set_color('orangered')
    bars[best_idx].set_alpha(1.0)

    ax.set_xlabel('EMG Strip Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy by EMG Strip', fontsize=14, fontweight='bold')
    ax.set_xticks(strip_numbers)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (strip, acc, std) in enumerate(zip(strip_numbers, mean_accs, std_accs)):
        ax.text(strip, acc + std + 0.02, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Other strips'),
        Patch(facecolor='orangered', alpha=1.0, label='Best strip')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{label}_strip_comparison_bars.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Bar plot saved: {plot_path}")
    plt.close()

    # ========================================================================
    # Plot 2: Line plot with confidence intervals
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(strip_numbers, mean_accs, 'o-', color='steelblue',
            linewidth=2, markersize=8, label='Mean accuracy')
    ax.fill_between(strip_numbers,
                    np.array(mean_accs) - np.array(std_accs),
                    np.array(mean_accs) + np.array(std_accs),
                    alpha=0.3, color='steelblue', label='±1 std dev')

    # Mark best strip
    best_strip_num = strip_numbers[best_idx]
    ax.plot(best_strip_num, mean_accs[best_idx], 'r*',
            markersize=20, label=f'Best (Strip {best_strip_num})')

    ax.set_xlabel('EMG Strip Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy Across EMG Strips', fontsize=14, fontweight='bold')
    ax.set_xticks(strip_numbers)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{label}_strip_comparison_line.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Line plot saved: {plot_path}")
    plt.close()

    # ========================================================================
    # Plot 3: Heatmap showing strip performance
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 3))

    # Create 1x10 heatmap
    data = np.array(mean_accs).reshape(1, -1)

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('Accuracy', fontsize=11, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(len(strip_numbers)))
    ax.set_xticklabels([f'Strip {s}' for s in strip_numbers], rotation=0)
    ax.set_yticks([])

    # Add text annotations
    for i, (acc, std) in enumerate(zip(mean_accs, std_accs)):
        text_color = 'white' if acc < 0.5 else 'black'
        ax.text(i, 0, f'{acc:.3f}\n±{std:.3f}',
                ha='center', va='center', fontsize=9,
                fontweight='bold', color=text_color)

    ax.set_title('EMG Strip Classification Performance Heatmap',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{label}_strip_comparison_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Heatmap saved: {plot_path}")
    plt.close()

    # ========================================================================
    # Plot 4: Box plot showing fold variability
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    fold_data = [results[s]['fold_accuracies'] for s in strip_numbers]

    bp = ax.boxplot(fold_data, positions=strip_numbers, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if i == best_idx:
            patch.set_facecolor('orangered')
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)

    ax.set_xlabel('EMG Strip Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy Distribution by EMG Strip (K-Fold Results)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(strip_numbers)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{label}_strip_comparison_boxplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Box plot saved: {plot_path}")
    plt.close()

    logging.info(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Test classification performance across EMG electrode strips"
    )
    p.add_argument("--root_dir", type=str, required=True,
                   help="Root directory containing dataset")
    p.add_argument("--label", type=str, required=True,
                   help="Dataset label (e.g., 'exo_gestures')")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for results (default: <root_dir>/strip_analysis)")
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Path to .npz dataset (default: <root_dir>/<label>_training_dataset.npz)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs per fold (default: 50)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--n_folds", type=int, default=5,
                   help="Number of CV folds (default: 5)")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging")

    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.root_dir, "strip_analysis")

    test_all_strips(
        root_dir=args.root_dir,
        label=args.label,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )