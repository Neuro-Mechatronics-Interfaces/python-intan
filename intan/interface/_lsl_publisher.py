# gesture_lsl.py
from __future__ import annotations
import json
from typing import Optional, Dict, Any

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock, cf_string
except ImportError as e:
    raise SystemExit(
        "pylsl is not installed. Install it with:\n\n    pip install pylsl\n"
    ) from e


class LSLPublisher:
    """
    Publish messages over Lab Streaming Layer (LSL) as a string Marker stream.

    - One channel (string), nominal_srate=0 (irregular), type='Markers' by default.
    - Call `publish(label)` to send 'pinkyflexion', 'thumbflexion', etc.
    - Optional: pass metadata to embed model/config info in the stream description.
    - Optional: de-duplicate consecutive identical labels.

    Example:
        lsl = GestureLSLBroadcaster(name="EMGGesture", metadata={"model":"128_channels"})
        lsl.publish("pinkyflexion")
        lsl.close()
    """

    def __init__(
        self,
        name: str = "Message",
        stream_type: str = "Markers",
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        only_on_change: bool = True,
        chunk_size: int = 1,
        max_buffered: int = 360,
    ) -> None:
        """
        Args:
            name: Stream name shown in LSL (e.g., "EMGGesture").
            stream_type: LSL type. "Markers" is commonly used for string events.
            source_id: Optional unique ID to distinguish multiple sources.
            metadata: Dict with any extra info (model label, classes, fs, etc.).
            only_on_change: If True, suppress duplicate consecutive labels.
            chunk_size: Outlet chunk size (usually 1 for markers).
            max_buffered: Max buffered samples in outlet.
        """
        self.name = name
        self.stream_type = stream_type
        self.source_id = source_id or f"{name}_source"
        self.only_on_change = bool(only_on_change)

        # One string channel, irregular sampling (0 Hz)
        info = StreamInfo(
            name=name,
            type=stream_type,
            channel_count=1,
            nominal_srate=0.0,        # Irregular/event stream
            channel_format=cf_string, # String marker channel
            source_id=self.source_id,
        )

        # Attach optional metadata to the stream description
        if metadata:
            desc = info.desc()
            for k, v in metadata.items():
                # Store non-str primitives directly; JSON-encode complex objects
                if isinstance(v, (str, int, float, bool)) or v is None:
                    desc.append_child_value(k, str(v))
                else:
                    desc.append_child_value(k, json.dumps(v, ensure_ascii=False))

        self._outlet = StreamOutlet(info, chunk_size=chunk_size, max_buffered=max_buffered)
        self._last_value: Optional[str] = None

    def publish(self, label: str, timestamp: Optional[float] = None) -> None:
        """
        Push a single gesture label as a string sample.

        Args:
            label: Your gesture string, e.g. "pinkyflexion".
            timestamp: LSL timestamp; if None, uses local_clock().
        """
        if not isinstance(label, str):
            label = str(label)

        if self.only_on_change and label == self._last_value:
            return

        ts = timestamp if (timestamp is not None) else local_clock()
        self._outlet.push_sample([label], ts)
        self._last_value = label

    def close(self) -> None:
        """Close the outlet (optional; GC will handle it too)."""
        try:
            # Just drop references so GC can clean up.
            self._outlet = None  # type: ignore
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
