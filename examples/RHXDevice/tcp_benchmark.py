#!/usr/bin/env python3
import argparse, socket, time
import numpy as np

from intan.interface import IntanRHXDevice
try:
    # your device module defines this; if not, fall back to 128
    from intan.interface import FRAMES_PER_BLOCK  # not actually needed here
except Exception:
    FRAMES_PER_BLOCK = 128

def human_mb(x):  # decimal MB/s for easy mental math
    return x / 1_000_000.0

def bench(duration=5.0, channels=None, blocks_per_write=None, read_bytes=262144, verbose=True):
    dev = IntanRHXDevice()
    try:
        # enable the channels you want active on TCP
        if channels is None:
            channels = range(dev.num_channels)     # or e.g. [15] to test one chan
        dev.enable_wide_channel(channels)

        # optional: increase blocks per write (RHX setting)
        if blocks_per_write is not None and hasattr(dev, "set_blocks_per_write"):
            dev.set_blocks_per_write(int(blocks_per_write))

        # socket options: large recv buffer + short timeout
        dev.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB
        dev.data_socket.settimeout(0.01)

        # start run mode
        dev.set_run_mode("run")
        fs = float(dev.sample_rate)  # Hz, per your config API
        n_active = len(list(channels))
        bytes_per_frame = 4 + 2 * n_active      # 4 bytes ts + 2 bytes per channel

        if verbose:
            print(f"[i] fs={fs:.1f} Hz, channels={n_active}, bytes/frame={bytes_per_frame}, "
                  f"theoretical ≈ {human_mb(fs*bytes_per_frame):.3f} MB/s")

        t0 = time.perf_counter()
        last = t0
        total = 0
        last_total = 0

        # small warm-up so first second doesn't look weird
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

            # per-second print
            if now - last >= 1.0 and verbose and now >= warm_until:
                sec_bytes = total - last_total
                sec_frames = sec_bytes / bytes_per_frame
                #print(f"[1s] {human_mb(sec_bytes):6.3f} MB/s  "
                #      f"≈ {sec_frames:8.0f} frames/s "
                #      f"(target ~{fs:.0f})")
                last = now
                last_total = total

        elapsed = time.perf_counter() - t0
        avg_bps = total / elapsed
        avg_frames = avg_bps / bytes_per_frame

        print("\n=== Summary ===")
        print(f"Elapsed: {elapsed:.3f}s")
        print(f"Received: {human_mb(total):.3f} MB total")
        print(f"Avg rate: {human_mb(avg_bps):.3f} MB/s "
              f"(est {avg_frames:.0f} frames/s; target ~{fs:.0f})")

    finally:
        # be nice to the app: stop streaming and close sockets
        try:
            dev.set_run_mode("stop")
        except Exception:
            pass
        dev.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Low-level RHX TCP throughput benchmark (no parsing).")
    p.add_argument("--sec", type=float, default=20.0, help="seconds to run")
    p.add_argument("--channels", type=str, default='0-63',
                   help="comma/range list (e.g. '0-127' or '15,16,17'). Default: all device.num_channels")
    p.add_argument("--blocks", type=int, default=None, help="TCPNumberDataBlocksPerWrite to set (optional)")
    p.add_argument("--read_bytes", type=int, default=262144, help="recv() read size in bytes")
    args = p.parse_args()

    # parse channels argument
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

    bench(duration=args.sec, channels=ch, blocks_per_write=args.blocks,
          read_bytes=args.read_bytes, verbose=True)
