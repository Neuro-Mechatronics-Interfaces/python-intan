#!/usr/bin/env python3
"""
Connect to Intan RHX software and stream EMG data.

This script demonstrates basic RHX device connection and streaming with optional
LSL output for real-time data sharing with other applications.

Usage:
    python rhx_connect_to_device.py --channels 128 --verbose
    python rhx_connect_to_device.py --use_lsl --channels 64 --channel_port b

The script will:
1. Connect to RHX software via TCP (ports 5000/5001 by default)
2. Configure the specified channels
3. Start streaming data to the internal buffer
4. Optionally publish data to LSL for external applications
"""
import argparse
import time
import sys
from intan.interface import IntanRHXDevice, LSLOptions


def main():
    """Main entry point for RHX device connection."""
    ap = argparse.ArgumentParser(
        description="Intan RHX Device Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--host", type=str, default="127.0.0.1", help="RHX server IP address")
    ap.add_argument("--command_port", type=int, default=5000, help="Command port")
    ap.add_argument("--data_port", type=int, default=5001, help="Data port")
    ap.add_argument("--channels", type=int, default=128, help="Number of channels")
    ap.add_argument("--channel_port", type=str, default="a", help="Channel port (a, b, c, d)")
    ap.add_argument("--sample_rate", type=float, default=None, help="Sample rate (Hz), if None, will query device")
    ap.add_argument("--buffer_duration", type=float, default=5.0, help="Buffer duration in seconds")
    ap.add_argument("--auto_start", action="store_true", help="Automatically start streaming on connect")
    ap.add_argument("--use_lsl", action="store_true", help="Enable LSL streaming")
    ap.add_argument("--lsl_numeric_name", type=str, default="EMG", help="LSL numeric stream name")
    ap.add_argument("--lsl_numeric_type", type=str, default="EMG", help="LSL numeric stream type")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = ap.parse_args()

    # Configure LSL options if requested
    lsl_opts = None
    if args.use_lsl:
        lsl_opts = LSLOptions(
            numeric_name=args.lsl_numeric_name,
            numeric_type=args.lsl_numeric_type,
            with_markers=False,
            chunk_size=32,
            source_id="IntanRHX_001",
        )

    # Use context manager for proper resource cleanup
    try:
        with IntanRHXDevice(
            host=args.host,
            command_port=args.command_port,
            data_port=args.data_port,
            num_channels=args.channels,
            sample_rate=args.sample_rate,
            buffer_duration_sec=args.buffer_duration,
            auto_start=False,  # We'll start after channel configuration
            use_lsl=args.use_lsl,
            lsl_options=lsl_opts,
            verbose=args.verbose
        ) as rhx:

            if not rhx.connected:
                print("[ERROR] Failed to connect to RHX device.")
                print("Checklist:")
                print("  1. Is RHX software running?")
                print("  2. Is TCP server enabled? (Network → TCP → Enable Server)")
                print(f"  3. Is the host reachable? (trying {args.host}:{args.command_port})")
                sys.exit(1)

            print(f"[OK] Connected to RHX device at {args.host}")
            print(f"     Channels: {args.channels}, Sample rate: {rhx.sample_rate:.1f} Hz")

            # Configure channels
            print(f"[SETUP] Configuring {args.channels} channels on port '{args.channel_port}'...")
            rhx.clear_all_data_outputs()
            rhx.enable_wide_channel(range(args.channels), port=args.channel_port)

            # Start streaming
            if not args.auto_start:
                rhx.start_streaming()
            
            if args.use_lsl:
                print(f"[LSL] Publishing to stream: '{args.lsl_numeric_name}' (type: {args.lsl_numeric_type})")
            
            print("[RUN] Streaming... Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[STOP] Stopping streaming...")
            
            # Context manager will handle cleanup (stop_streaming, close)
            print("[DONE] Disconnected.")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

