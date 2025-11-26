"""
Optimized Intan I/O module with data compression and progress tracking.

Key improvements:
1. Compressed NPZ format (uses numpy.savez_compressed)
2. Memory-mapped loading for large files
3. Progress bars with tqdm
4. Data type optimization (use int16 instead of float64 where appropriate)
"""

import numpy as np
import os
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    # Fallback: simple progress indicator
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else None
            self.total = kwargs.get('total', None)
            self.desc = kwargs.get('desc', '')
            self.disable = kwargs.get('disable', False)
            
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass
        
        def set_description(self, desc):
            self.desc = desc
        
        def close(self):
            pass


def save_as_npz_compressed(
    result: Dict[str, Any], 
    output_path: str,
    show_progress: bool = True,
    optimize_dtypes: bool = True
) -> Dict[str, str]:
    """
    Save Intan data to compressed NPZ format with significant size reduction.
    
    Parameters:
    -----------
    result : dict
        Dictionary containing Intan recording data
    output_path : str
        Path where the NPZ file will be saved
    show_progress : bool
        Whether to show progress bar (requires tqdm)
    optimize_dtypes : bool
        Whether to optimize data types to reduce file size
        
    Returns:
    --------
    dict : Information about the saved file including compression ratio
    
    Optimization strategies:
    - Use savez_compressed instead of savez (applies gzip compression)
    - Convert float64 to float32 where full precision isn't needed
    - Keep integer data as int16 (ADC data is typically 16-bit)
    - Compress string data efficiently
    """
    
    print(f"Saving compressed NPZ file to: {output_path}")
    
    # Prepare data dictionary with optimized types
    save_dict = {}
    original_size = 0
    optimized_size = 0
    
    # Progress bar setup
    items = list(result.items())
    pbar = tqdm(items, desc="Processing data fields", disable=not show_progress or not TQDM_AVAILABLE)
    
    for key, value in pbar:
        pbar.set_description(f"Processing: {key[:30]}")
        
        if isinstance(value, np.ndarray):
            original_size += value.nbytes
            
            if optimize_dtypes:
                # Optimization logic based on data type and key
                if value.dtype == np.float64:
                    # Time vectors and non-ADC data can often use float32
                    if 't_' in key.lower() or 'time' in key.lower():
                        value = value.astype(np.float32)
                    # ADC data might be stored as float but originated as int16
                    elif 'amplifier' in key.lower() or 'aux' in key.lower():
                        # Check if data is in integer range
                        if np.all(np.abs(value) < 32767):
                            # Scale to microvolts if needed, keep as int16
                            value = value.astype(np.int16)
                        else:
                            value = value.astype(np.float32)
                    else:
                        value = value.astype(np.float32)
                
                elif value.dtype == np.int64:
                    # Most integer data doesn't need 64 bits
                    if np.max(np.abs(value)) < 32767:
                        value = value.astype(np.int16)
                    elif np.max(np.abs(value)) < 2147483647:
                        value = value.astype(np.int32)
            
            optimized_size += value.nbytes
            save_dict[key] = value
        else:
            # Non-array data (strings, scalars, etc.)
            save_dict[key] = value
    
    pbar.close()
    
    # Save with compression
    print(f"\nCompressing and saving...")
    print(f"  Original data size: {original_size / 1024**2:.2f} MB")
    print(f"  Optimized data size: {optimized_size / 1024**2:.2f} MB")
    print(f"  Memory reduction: {(1 - optimized_size/original_size)*100:.1f}%")
    
    # Use savez_compressed for automatic gzip compression
    np.savez_compressed(output_path, **save_dict)
    
    # Get actual file size
    file_size = os.path.getsize(output_path)
    compression_ratio = original_size / file_size if file_size > 0 else 0
    
    print(f"  Compressed file size: {file_size / 1024**2:.2f} MB")
    print(f"  Overall compression ratio: {compression_ratio:.2f}x")
    print(f"  Space saved: {(1 - file_size/original_size)*100:.1f}%")
    
    return {
        'output_path': output_path,
        'original_size_mb': original_size / 1024**2,
        'optimized_size_mb': optimized_size / 1024**2,
        'compressed_size_mb': file_size / 1024**2,
        'compression_ratio': compression_ratio,
        'space_saved_percent': (1 - file_size/original_size) * 100
    }


def load_npz_file(
    npz_path: str,
    show_progress: bool = True,
    mmap_mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load NPZ file with progress tracking.
    
    Parameters:
    -----------
    npz_path : str
        Path to the NPZ file
    show_progress : bool
        Whether to show progress bar
    mmap_mode : str, optional
        Memory-map mode ('r', 'r+', 'w+', 'c'). Use 'r' for read-only memory mapping
        of large files to avoid loading everything into RAM.
        
    Returns:
    --------
    dict : Dictionary containing all the loaded data
    
    Note: For very large files, consider using mmap_mode='r' to memory-map
    the arrays instead of loading them entirely into memory.
    """
    
    print(f"Loading NPZ file: {npz_path}")
    file_size = os.path.getsize(npz_path)
    print(f"  File size: {file_size / 1024**2:.2f} MB")
    
    # Load the NPZ file
    if mmap_mode:
        print(f"  Using memory-mapped mode: {mmap_mode}")
        npz_file = np.load(npz_path, mmap_mode=mmap_mode, allow_pickle=True)
    else:
        npz_file = np.load(npz_path, allow_pickle=True)
    
    # Get list of arrays
    array_keys = list(npz_file.keys())
    
    # Load arrays with progress bar
    result = {}
    pbar = tqdm(array_keys, desc="Loading arrays", disable=not show_progress or not TQDM_AVAILABLE)
    
    for key in pbar:
        pbar.set_description(f"Loading: {key[:30]}")
        
        if mmap_mode:
            # Keep as memory-mapped array
            result[key] = npz_file[key]
        else:
            # Load into memory
            result[key] = npz_file[key]
            
            # Handle scalar/0-d arrays
            if isinstance(result[key], np.ndarray) and result[key].ndim == 0:
                result[key] = result[key].item()
    
    pbar.close()
    
    print(f"✓ Loaded {len(result)} fields from NPZ file")
    
    # Print summary of loaded data
    print("\nData summary:")
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"size={value.nbytes/1024**2:.2f} MB")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    return result


def compare_file_sizes(dat_dir: str, npz_path: str) -> None:
    """
    Compare sizes of original DAT files with the NPZ file.
    
    Parameters:
    -----------
    dat_dir : str
        Directory containing the original .dat files
    npz_path : str
        Path to the NPZ file
    """
    print("\n" + "="*60)
    print("FILE SIZE COMPARISON")
    print("="*60)
    
    # Get DAT file sizes
    dat_files = list(Path(dat_dir).glob('*.dat'))
    total_dat_size = sum(f.stat().st_size for f in dat_files)
    
    print(f"\nOriginal DAT files ({len(dat_files)} files):")
    for f in dat_files:
        size_mb = f.stat().st_size / 1024**2
        print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  TOTAL: {total_dat_size / 1024**2:.2f} MB")
    
    # Get NPZ file size
    npz_size = os.path.getsize(npz_path)
    print(f"\nCompressed NPZ file:")
    print(f"  {Path(npz_path).name}: {npz_size / 1024**2:.2f} MB")
    
    # Calculate ratio
    ratio = npz_size / total_dat_size
    print(f"\nCompression result:")
    if ratio < 1:
        print(f"  ✓ NPZ is {(1-ratio)*100:.1f}% SMALLER than original")
        print(f"  ✓ Compression ratio: {1/ratio:.2f}x")
    else:
        print(f"  ✗ NPZ is {(ratio-1)*100:.1f}% LARGER than original")
        print(f"  ✗ Expansion ratio: {ratio:.2f}x")
    
    print("="*60)


# Example usage
if __name__ == "__main__":
    """
    Example demonstrating the optimized save and load functions.
    """
    
    # This is a placeholder - in practice, you'd load actual data
    print("Example: Optimized NPZ saving and loading")
    print("-" * 60)
    
    # Simulate some data (in practice, this comes from load_dat_file)
    dummy_data = {
        'amplifier_data': np.random.randint(-5000, 5000, (64, 100000), dtype=np.int16),
        't_amplifier': np.linspace(0, 100, 100000, dtype=np.float64),
        'sampling_rate': np.array(30000.0),
        'export_basename': 'test_recording',
        'export_basepath': '/path/to/data'
    }
    
    # Save with compression
    output_file = '/tmp/test_compressed.npz'
    stats = save_as_npz_compressed(dummy_data, output_file)
    
    print("\n" + "="*60)
    print("LOADING DEMONSTRATION")
    print("="*60)
    
    # Load with progress
    loaded_data = load_npz_with_progress(output_file)
    
    # Verify data integrity
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)
    if np.array_equal(dummy_data['amplifier_data'], loaded_data['amplifier_data']):
        print("✓ Data integrity verified: Arrays match!")
    else:
        print("✗ Warning: Data mismatch detected")
