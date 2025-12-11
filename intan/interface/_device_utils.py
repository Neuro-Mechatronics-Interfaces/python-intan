"""
Utility functions for Intan device channel management and configuration.
"""

import re
import logging
from collections import defaultdict
from typing import List, Dict, Tuple


# Regex for parsing channel names like 'B-000', 'b_17', 'A-003'
CHANNEL_NAME_PATTERN = re.compile(r'^\s*([A-Da-d])\s*[-_ ]?\s*(\d{1,3})\s*$')


def normalize_channel_names(names: List[str]) -> Tuple[List[str], bool]:
    """
    Normalize channel names to canonical 1-based format (e.g., 'A-001', 'B-032').
    
    If any channel '000' is detected, treats the entire set as 0-indexed and shifts by +1.
    
    Args:
        names: List of channel names (e.g., ['B-000', 'B-001', 'A-017'])
        
    Returns:
        Tuple of (normalized_names, was_zero_indexed)
        
    Example:
        >>> normalize_channel_names(['A-000', 'A-001', 'B-000'])
        (['A-001', 'A-002', 'B-001'], True)
        
        >>> normalize_channel_names(['A-001', 'A-002', 'B-001'])
        (['A-001', 'A-002', 'B-001'], False)
    """
    parsed = []
    saw_zero = False
    
    for nm in names:
        m = CHANNEL_NAME_PATTERN.match(str(nm))
        if not m:
            continue
        port = m.group(1).upper()
        num = int(m.group(2))
        if num == 0:
            saw_zero = True
        parsed.append((port, num))
    
    # If any zero present, interpret the whole set as zero-indexed
    shift = 1 if saw_zero else 0
    
    canon = []
    for port, num in parsed:
        n1 = num + shift
        canon.append(f"{port}-{n1:03d}")
    
    return canon, saw_zero


def parse_channel_spec(channel_names: List[str]) -> Dict[str, List[int]]:
    """
    Parse channel names into port-indexed dictionary.
    
    Args:
        channel_names: List of canonical channel names (e.g., ['A-001', 'B-032'])
        
    Returns:
        Dictionary mapping port letter to list of 0-based indices
        
    Example:
        >>> parse_channel_spec(['A-001', 'A-003', 'B-001'])
        {'a': [0, 2], 'b': [0]}
    """
    by_port = defaultdict(list)
    
    for nm in channel_names:
        m = CHANNEL_NAME_PATTERN.match(str(nm))
        if not m:
            logging.warning(f"Skipping unrecognized channel name: {nm!r}")
            continue
        
        port = m.group(1).lower()  # 'a', 'b', 'c', 'd'
        ch_1b = int(m.group(2))    # 1-based index
        idx0 = ch_1b - 1           # Convert to 0-based
        
        if 0 <= idx0 < 128:
            by_port[port].append(idx0)
        else:
            logging.warning(f"Out-of-range channel {nm} -> {port.upper()}-{ch_1b:03d}")
    
    # Deduplicate and sort
    for port in by_port:
        by_port[port] = sorted(set(by_port[port]))
    
    return dict(by_port)


def enable_channels_by_name(device, channel_names: List[str]) -> Dict[str, List[int]]:
    """
    Enable Intan device channels based on canonical channel names.
    
    Args:
        device: IntanRHXDevice instance
        channel_names: List of canonical channel names (e.g., ['A-001', 'B-032'])
        
    Returns:
        Dictionary mapping port to list of enabled 0-based indices
        
    Example:
        >>> enabled = enable_channels_by_name(dev, ['A-001', 'A-002', 'B-001'])
        >>> enabled
        {'a': [0, 1], 'b': [0]}
    """
    by_port = parse_channel_spec(channel_names)
    
    for port, indices in by_port.items():
        try:
            # Try vector form if supported
            device.enable_wide_channel(indices, port=port)
        except Exception:
            # Fallback to scalar form
            for i in indices:
                device.enable_wide_channel(i, port=port)
    
    logging.info(
        "Enabled channels: " + ", ".join(
            f"{p.upper()}:{len(idxs)}" for p, idxs in sorted(by_port.items())
        )
    )
    
    return by_port


def get_device_channel_names(device) -> List[str]:
    """
    Get channel names from device, with fallback to generic names.
    
    Args:
        device: IntanRHXDevice instance
        
    Returns:
        List of channel names
    """
    if hasattr(device, "get_channel_names"):
        try:
            names = list(device.get_channel_names())
            if names:
                return names
        except Exception:
            pass
    
    # Fallback to generic names
    num_channels = getattr(device, "num_channels", 128)
    return [f"CH{i}" for i in range(num_channels)]


def build_active_channel_order(enabled_by_port: Dict[str, List[int]]) -> List[str]:
    """
    Build the ordered list of channel names as they will appear in streamed data.
    
    Args:
        enabled_by_port: Dictionary from enable_channels_by_name()
        
    Returns:
        List of canonical channel names in stream order
        
    Example:
        >>> build_active_channel_order({'a': [0, 2], 'b': [1]})
        ['A-001', 'A-003', 'B-002']
    """
    active_names = []
    for port in sorted(enabled_by_port.keys()):  # Deterministic ordering
        for idx0 in enabled_by_port[port]:
            active_names.append(f"{port.upper()}-{idx0 + 1:03d}")
    return active_names
