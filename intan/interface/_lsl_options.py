from typing import Optional, Sequence
from dataclasses import dataclass

@dataclass
class LSLOptions:
    numeric_name: str = "EMG-Device"
    numeric_type: str = "EMG"
    source_id:   str = "intan-001"
    fs:          float = 2000.0      # nominal; override at connect if you know the exact rate
    channels:    int = 128
    chunk_size:  int = 32            # samples per push
    with_markers: bool = True
    marker_name: str = "EMG-Markers"
    marker_type: str = "Markers"
    channel_labels: Optional[Sequence[str]] = None
