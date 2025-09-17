import argparse
import time
from intan.interface import IntanRHXDevice, LSLOptions


# Test script if main, just connects to teh device and pushes data to LSL, no need to record
if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Intan RHX Device Interface")
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

    # parse args
    args = ap.parse_args()

    # Make RHX device client
    rhx = IntanRHXDevice(
        host=args.host,
        command_port=args.command_port,
        data_port=args.data_port,
        num_channels=args.channels,
        sample_rate=args.sample_rate,
        buffer_duration_sec=args.buffer_duration,
        auto_start=args.auto_start,
        use_lsl=args.use_lsl,
        lsl_options=LSLOptions(
            numeric_name=args.lsl_numeric_name,
            numeric_type=args.lsl_numeric_type,
            with_markers=False,
            chunk_size=32,
            source_id="IntanRHX_001",
        ),
        verbose=args.verbose
    )


    if not rhx.connected:
        print("Failed to connect to RHX device.")
    else:
        print("Connected to RHX device.")
        # Enable channels
        rhx.clear_all_data_outputs()
        for ch in range(args.channels):
            rhx.enable_wide_channel(ch, port=args.channel_port)

        if not args.auto_start:
            rhx.start_streaming()
        print("Streaming... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        rhx.stop_streaming()
        print("Stopped.")

