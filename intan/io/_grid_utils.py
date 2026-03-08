"""
intan.io._grid_utils

Utilities for working with high-density EMG electrode arrays and grids.

This module provides functions for:
- Inferring grid dimensions from channel naming conventions
- Applying grid permutations (rotations, flips) for different orientations
- Parsing orientation tags from filenames
- Remapping channels between different grid layouts

Typical use case: HD-EMG electrode arrays where the physical orientation
of the array on the limb may vary between recordings, requiring consistent
channel ordering for machine learning.
"""

import os
import re
from typing import List, Tuple, Optional
import numpy as np


def infer_grid_dimensions(channel_names) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer grid dimensions (rows, cols) from channel naming conventions.
    
    Supports naming patterns like:
        - "8-8-L" in mapping name → 8x8 grid
        - List of 64 channels → could be 8x8
        - Sequential numbers with clear row/column structure
    
    Args:
        channel_names: List of channel names or a mapping name string
    
    Returns:
        Tuple of (n_rows, n_cols) if inferrable, else (None, None)
    
    Example:
        >>> infer_grid_dimensions(["A-001", "A-002", ..., "A-064"])
        (8, 8)
    """
    if not channel_names:
        return None, None
    
    # If passed a single string (mapping name), check for dimension pattern
    if isinstance(channel_names, str):
        # Pattern like "8-8-L", "8x8", "16-8", etc.
        match = re.search(r'(\d+)[-x](\d+)', channel_names)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    n = len(channel_names)
    
    # Try to find a square grid first
    sqrt_n = int(np.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return sqrt_n, sqrt_n
    
    # Try common rectangular grids
    common_grids = [
        (16, 8), (8, 16),  # 128 channels
        (8, 4), (4, 8),    # 32 channels
        (16, 4), (4, 16),  # 64 channels
        (12, 8), (8, 12),  # 96 channels
    ]
    
    for rows, cols in common_grids:
        if rows * cols == n:
            return rows, cols
    
    # Factor-based fallback
    for rows in range(2, int(np.sqrt(n)) + 1):
        if n % rows == 0:
            cols = n // rows
            if cols >= 2:
                return rows, cols
    
    return None, None


def apply_grid_permutation(
    indices: List[int],
    n_rows: int,
    n_cols: int,
    mode: str,
) -> List[int]:
    """
    Apply a spatial permutation to grid channel indices.
    
    Useful for handling different physical orientations of electrode arrays.
    The indices are treated as a row-major flattened grid.
    
    Args:
        indices: Original channel indices (0-based, row-major order)
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        mode: Transformation to apply:
            - "none" or "N": No transformation
            - "rot90" or "R": 90° clockwise rotation
            - "rot180": 180° rotation
            - "rot270": 270° clockwise rotation (same as 90° counter-clockwise)
            - "flipH" or "H": Horizontal flip (left-right)
            - "flipV" or "V": Vertical flip (top-bottom)
            - "transpose" or "T": Transpose (swap rows/cols)
    
    Returns:
        Permuted indices
    
    Example:
        >>> # 2x3 grid: [[0,1,2], [3,4,5]] rotated 90° CW → [[3,0], [4,1], [5,2]]
        >>> apply_grid_permutation([0,1,2,3,4,5], 2, 3, "rot90")
        [3, 0, 4, 1, 5, 2]
    """
    mode = mode.lower().strip()
    
    if mode in ("none", "n", ""):
        return list(indices)
    
    n = len(indices)
    expected = n_rows * n_cols
    
    if n != expected:
        raise ValueError(f"Index count ({n}) doesn't match grid size ({n_rows}x{n_cols}={expected})")
    
    # Create 2D grid
    grid = np.array(indices).reshape(n_rows, n_cols)
    
    # Apply transformation
    if mode in ("rot90", "r"):
        grid = np.rot90(grid, k=-1)  # CW
    elif mode == "rot180":
        grid = np.rot90(grid, k=2)
    elif mode in ("rot270", "ccw90"):
        grid = np.rot90(grid, k=1)  # CCW = 270 CW
    elif mode in ("fliph", "h", "fliplr"):
        grid = np.fliplr(grid)
    elif mode in ("flipv", "v", "flipud"):
        grid = np.flipud(grid)
    elif mode in ("transpose", "t"):
        grid = grid.T
    else:
        raise ValueError(f"Unknown permutation mode: {mode}")
    
    return grid.flatten().tolist()


def parse_orientation_from_filename(path: str) -> Optional[str]:
    """
    Extract orientation tag from a filename.
    
    Looks for common orientation indicators in the filename:
        - "CCW90", "CW90", "ROT90", "ROT180", "ROT270"
        - "FLIPH", "FLIPV", "FLIPPED"
        - Single letters at end: "_R", "_L", "_N"
    
    Args:
        path: File path or filename
    
    Returns:
        Orientation string if found, else None
    
    Example:
        >>> parse_orientation_from_filename("trial1_CCW90_241112.npz")
        'CCW90'
        >>> parse_orientation_from_filename("recording_R.rhd")
        'R'
    """
    name = os.path.splitext(os.path.basename(str(path)))[0].upper()
    
    # Explicit rotation tags
    patterns = [
        (r'CCW\s*90', 'CCW90'),
        (r'CW\s*90', 'CW90'),
        (r'ROT\s*90', 'ROT90'),
        (r'ROT\s*180', 'ROT180'),
        (r'ROT\s*270', 'ROT270'),
        (r'FLIP\s*H', 'FLIPH'),
        (r'FLIP\s*V', 'FLIPV'),
        (r'FLIPPED', 'FLIPPED'),
    ]
    
    for pattern, tag in patterns:
        if re.search(pattern, name):
            return tag
    
    # Single letter suffix: _R, _L, _N
    match = re.search(r'[_-]([RLNTV])(?:[_-]|$)', name)
    if match:
        return match.group(1)
    
    return None


def orientation_to_permutation_mode(orientation: str) -> str:
    """
    Convert an orientation tag to a permutation mode string.
    
    Args:
        orientation: Orientation tag (e.g., "CCW90", "R", "FLIPH")
    
    Returns:
        Permutation mode for apply_grid_permutation()
    
    Example:
        >>> orientation_to_permutation_mode("CCW90")
        'rot270'
        >>> orientation_to_permutation_mode("R")
        'rot90'
    """
    if not orientation:
        return "none"
    
    orientation = orientation.upper().strip()
    
    mapping = {
        # Explicit rotations
        "CCW90": "rot270",
        "CW90": "rot90",
        "ROT90": "rot90",
        "ROT180": "rot180",
        "ROT270": "rot270",
        # Flips
        "FLIPH": "flipH",
        "FLIPV": "flipV",
        "FLIPPED": "rot180",  # Common meaning
        # Single letters (convention-dependent)
        "R": "rot90",      # Right/rotated
        "L": "rot270",     # Left
        "N": "none",       # Normal/neutral
        "T": "transpose",
        "V": "flipV",
        "H": "flipH",
    }
    
    return mapping.get(orientation, "none")


def remap_grid_channels(
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
    orientation: str,
) -> np.ndarray:
    """
    Remap channels in a data array according to grid orientation.
    
    Convenience function that combines orientation parsing and permutation.
    
    Args:
        data: EMG data array, shape (n_channels, n_samples)
        n_rows: Grid rows
        n_cols: Grid columns
        orientation: Orientation tag or mode string
    
    Returns:
        Remapped data array
    
    Example:
        >>> emg = np.random.randn(64, 1000)  # 8x8 grid
        >>> remapped = remap_grid_channels(emg, 8, 8, "CCW90")
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, samples), got shape {data.shape}")
    
    n_channels = data.shape[0]
    expected = n_rows * n_cols
    
    if n_channels != expected:
        raise ValueError(f"Channel count ({n_channels}) doesn't match grid ({n_rows}x{n_cols})")
    
    mode = orientation_to_permutation_mode(orientation)
    
    if mode == "none":
        return data
    
    original_indices = list(range(n_channels))
    new_indices = apply_grid_permutation(original_indices, n_rows, n_cols, mode)
    
    return data[new_indices, :]