#!/usr/bin/env python3
"""
Low-level TCP throughput benchmark for RHX streaming.

Measures raw socket performance without parsing overhead to assess
maximum achievable data rates and identify bottlenecks.

Usage:
    python tcp_benchmark.py --sec 20 --channels 0-127
    python tcp_benchmark.py --sec 10 --channels 0-63 --blocks 8
"""
import argparse
import socket
import time
import sys
from intan.interface import IntanRHXDevice

try:
    from intan.interface import FRAMES_PER_BLOCK
except ImportError:
    FRAMES_PER_BLOCK = 128  # Fallback


def human_mb(x):
    """Convert bytes to decimal MB for easy readability."""
    return x / 1_000_000.0

def bench(duration=5.0, channels=None, blocks_per_write=None, read_bytes=262144, verbose=True):
    """Run TCP throughput benchmark."""
    print("[INIT] Connecting to RHX device...")
    dev = IntanRHXDevice()
    
    if not dev.connected:
        print("[ERROR] Failed to connect to RHX device.")
        print("Ensure RHX software is running with TCP server enabled.")
        return 1
    
    print("[OK] Connected.")
    
    try:
        # Enable channels
        if channels is None:
            channels = range(dev.num_channels)
        dev.enable_wide_channel(channels)

        # Optional: increase blocks per write (RHX setting)
        if blocks_per_write is not None and hasattr(dev, "set_blocks_per_write"):
            dev.set_blocks_per_write(int(blocks_per_write))

        # Socket options: large recv buffer + short timeout
        dev.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB
        dev.data_socket.settimeout(0.01)

        # start run mode
        dev.set_run_mode("run")
        fs = float(dev.sample_rate)  # Hz, per your config API
        n_active = len(list(channels))
        bytes_per_frame = 4 + 2 * n_active      # 4 bytes ts + 2 bytes per channel

        if verbose:
            print(f"[BENCH] Sample rate: {fs:.1f} Hz")
            print(f"[BENCH] Active channels: {n_active}")
            print(f"[BENCH] Bytes/frame: {bytes_per_frame}")
            print(f"[BENCH] Theoretical: ~{human_mb(fs*bytes_per_frame):.3f} MB/s")
            print(f"[RUN] Running benchmark for {duration:.1f}s...")

        t0 = time.perf_counter()
        last = t0
        total = 0
        last_total = 0

        # Small warm-up period
        warm_until = t0 + 0.2

        while True:
            now = time.perf_counter()
            if (now - t0) >= duration:
                break
            # drain the socket; request a big-ish chunk each call
            try:
                chunk = dev.data_socket.recv(read_bytes)
                if chunk:
                    total += len(chunk)
            except socket.timeout:
                pass

            # Per-second interval reporting
            if now - last >= 1.0 and verbose and now >= warm_until:
                sec_bytes = total - last_total
                sec_frames = sec_bytes / bytes_per_frame
                print(f"  {now-t0:6.1f}s → {human_mb(total):9.3f} MB total, "
                      f"{human_mb(sec_bytes):6.3f} MB/s interval")
                last = now
                last_total = total

        elapsed = time.perf_counter() - t0
        avg_bps = total / elapsed
        avg_frames = avg_bps / bytes_per_frame
        theoretical = human_mb(fs * bytes_per_frame)
        efficiency = 100 * (avg_bps / (fs * bytes_per_frame))

        print(f"\n{'='*60}")
        print(f"[SUMMARY] Benchmark Results")
        print(f"{'='*60}")
        print(f"Duration:        {elapsed:.2f}s")
        print(f"Total received:  {human_mb(total):.3f} MB")
        print(f"Average rate:    {human_mb(avg_bps):.3f} MB/s (~{avg_frames:.0f} frames/s)")
        print(f"Theoretical:     ~{theoretical:.3f} MB/s (~{fs:.0f} frames/s)")
        print(f"Efficiency:      {efficiency:.1f}%")
        print(f"{'='*60}")

    finally:
        # be nice to the app: stop streaming and close sockets
        try:
            dev.set_run_mode("stop")
        except Exception:
            pass
        dev.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Low-level RHX TCP throughput benchmark (no parsing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tcp_benchmark.py                          # 20s, channels 0-63
  python tcp_benchmark.py --sec 30 --channels 0-127
  python tcp_benchmark.py --blocks 8 --read_bytes 524288
        """
    )
    p.add_argument("--sec", type=float, default=20.0, 
                   help="Duration in seconds (default: 20.0)")
    p.add_argument("--channels", type=str, default='0-63',
                   help="Comma/range list (e.g. '0-127' or '15,16,17')")
    p.add_argument("--blocks", type=int, default=None, 
                   help="TCPNumberDataBlocksPerWrite (optional)")
    p.add_argument("--read_bytes", type=int, default=262144, 
                   help="recv() buffer size in bytes (default: 262144)")
    args = p.parse_args()

    # Parse channels argument
    ch = None
    if args.channels:
        ch = []
        for tok in args.channels.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                ch.extend(range(int(a), int(b) + 1))
            else:
                ch.append(int(tok))

    try:
        bench(duration=args.sec, channels=ch, blocks_per_write=args.blocks,
              read_bytes=args.read_bytes, verbose=True)
    except KeyboardInterrupt:
        print("\n[ABORTED] Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
