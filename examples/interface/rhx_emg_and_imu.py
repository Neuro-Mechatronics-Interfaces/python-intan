#!/usr/bin/env python3
"""
Log synchronized EMG (from RHX) + IMU (from Pico) to CSV file.

This script demonstrates multi-modal data acquisition by combining:
- High-rate EMG data from Intan RHX device (e.g., 1-4 kHz)
- IMU orientation/acceleration from Raspberry Pi Pico (typically ~100 Hz)

Output CSV format (comma-separated):
  EMG_0, EMG_1, ..., EMG_{C-1}, IMU_seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz, t_host

Each row represents one EMG sample with the most recent IMU data (no interpolation).
The timestamp t_host is from the host computer's clock (time.time()).

Usage:
    python rhx_emg_and_imu.py --channels 32 --outfile test.csv --verbose
    python rhx_emg_and_imu.py --channels 128 --channel_port a --flush_every 10000
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

# Intan RHX device
from intan.interface import IntanRHXDevice

# Pico IMU client (auto-discovery, single-socket)
from pico_imu_client import PicoIMUClient



def write_header(writer, n_channels: int):
    emg_cols = [f"EMG_{i}" for i in range(n_channels)]
    imu_cols = ["IMU_seq", "roll", "pitch", "yaw", "ax", "ay", "az", "gx", "gy", "gz"]
    writer.writerow(emg_cols + imu_cols + ["t_host"])


def main():
    ap = argparse.ArgumentParser(
        description="Log synchronized EMG + IMU data to CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--host", type=str, default="127.0.0.1", help="RHX server IP")
    ap.add_argument("--command_port", type=int, default=5000, help="RHX command port")
    ap.add_argument("--data_port", type=int, default=5001, help="RHX data port")
    ap.add_argument("--channels", type=int, default=32, help="Number of wide channels to enable")
    ap.add_argument("--channel_port", type=str, default="a", help="RHX front-end port (a/b/c/d)")
    ap.add_argument("--buffer_sec", type=float, default=5.0, help="RHX ring buffer duration (s)")
    ap.add_argument("--outfile", type=str, default="emg_imu.csv", help="Output CSV filename")
    ap.add_argument("--flush_every", type=int, default=5000, help="Flush file every N rows")
    ap.add_argument("--verbose", action="store_true", help="Print debug info")
    args = ap.parse_args()

    # ---- Connect RHX using context manager ----
    print(f"[RHX] Connecting to {args.host}:{args.command_port}...")
    try:
        rhx = IntanRHXDevice(
            host=args.host,
            command_port=args.command_port,
            data_port=args.data_port,
            num_channels=args.channels,
            sample_rate=None,  # Auto-detect
            buffer_duration_sec=args.buffer_sec,
            auto_start=False,
            use_lsl=False,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"[RHX][ERROR] Failed to connect: {e}")
        print("Checklist:")
        print("  1. Is RHX software running?")
        print("  2. Is TCP server enabled in RHX?")
        print(f"  3. Is {args.host}:{args.command_port} reachable?")
        return 1

    if not rhx.connected:
        print("[RHX][ERROR] Connection failed.")
        return 1

    # Configure channels
    print(f"[RHX] Configuring {args.channels} channels on port '{args.channel_port}'...")
    rhx.clear_all_data_outputs()
    rhx.enable_wide_channel(range(args.channels), port=args.channel_port)
    rhx.start_streaming()

    # Verify data is flowing
    print("[RHX] Probing for incoming samples...")
    probe_t0 = time.time()
    probe_seen = 0
    last_idx = rhx.circular_idx
        
    while time.time() - probe_t0 < 3.0:
        time.sleep(0.05)
        with rhx.buffer_lock:
            cur_idx = rhx.circular_idx
        if cur_idx != last_idx:
            probe_seen += (cur_idx - last_idx) % rhx.circular_buffer.shape[1]
            last_idx = cur_idx
        print(f"  idx={cur_idx} (+{probe_seen} new)", end="\r")

    print()
    if probe_seen == 0:
        print("[RHX][ERROR] No samples arrived in 3s.")
        print("Checklist:")
        print("  1. Is RHX in 'Run' mode (not 'Stop')?")
        print("  2. Are channels properly connected?")
        print("  3. Check RHX sample rate settings")
        rhx.stop_streaming()
        return 1
    print(f"[RHX][OK] Received ~{probe_seen} samples during probe.")

    fs = float(rhx.sample_rate)
    n_channels = int(rhx.num_channels)
    print(f"[RHX] Streaming {fs:.1f} Hz, {n_channels} channels")

    # ---- Start IMU ----
    print("[IMU] Connecting to Pico...")
    imu = PicoIMUClient(print_rate_hz=0.0)
    imu_ok = imu.start(timeout=6.0)
        
    if not imu_ok:
        print(f"[IMU][WARN] Discovery failed: {imu.last_error() or 'unknown'}")
        print("[IMU][WARN] Continuing without IMU (columns will be NaN)")
    else:
        print(f"[IMU][OK] Connected at ~{imu.rate_hz():.1f} Hz")

    # ---- Open CSV file ----
    try:
        fout = open(args.outfile, "w", newline="")
        writer = csv.writer(fout, delimiter=",")
        write_header(writer, n_channels)
        rows_written = 1  # header row
        print(f"[LOG] Writing to {os.path.abspath(args.outfile)}")
    except OSError as e:
        rhx.stop_streaming()
        print(f"[ERROR] Failed to open output file: {e}")
        return 1

    # ---- Drain RHX circular buffer to file ----
    buf = rhx.circular_buffer           # shape (C, L)
    L = buf.shape[1]
    prev_idx = rhx.circular_idx
    rows_since_flush = 0
    t0 = time.time()
    last_beat = time.time()
    last_rows = 0

    try:
        print("[RUN] Logging… Ctrl+C to stop.")
        while True:
            time.sleep(0.01)  # light backoff; we’re eventless-polling

            # snapshot new sample range under lock
            with rhx.buffer_lock:
                idx = rhx.circular_idx
                # compute how many NEW sample columns arrived since prev_idx
                if idx >= prev_idx:
                    n_new = idx - prev_idx
                    # contiguous slice: prev_idx .. idx-1
                    if n_new > 0:
                        # shape (C, n_new)
                        frame = buf[:, prev_idx:idx]
                else:
                    n_new = (L - prev_idx) + idx
                    if n_new > 0:
                        # wrap-around: concatenate (prev_idx .. L-1) + (0 .. idx-1)
                        part1 = buf[:, prev_idx:L]
                        part2 = buf[:, 0:idx]
                        # avoid numpy import; write in two passes below
                        frame = None  # we'll use two passes
                prev_idx = idx

            if n_new <= 0:
                continue

            # Get latest IMU (or NaNs)
            latest = imu.get_latest() if imu_ok else None
            if latest:
                imu_seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz = latest
            else:
                imu_seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz = (float("nan"),)*10
                
            now = time.time()
            if now - last_beat >= 1.0:
                if latest:
                    seq, r, p, y, ax, ay, az, gx, gy, gz = latest
                    print(f"[IMU] seq={seq} R={r:+6.2f} P={p:+6.2f} Y={y:+6.2f} "
                          f"Acc=({ax:+5.2f},{ay:+5.2f},{az:+5.2f}) Gy=({gx:+5.2f},{gy:+5.2f},{gz:+5.2f})")
                else:
                    print("[IMU] (no packets yet)")

                # rows written since last beat (after writerows below we’ll update writer.line_num)
                print(f"[LOG] rows so far={rows_written}")
                last_beat = now

            t_write = time.time()

            # Write rows:
            # For performance, batch-build rows and writerows() once.
            batch_rows = []

            if frame is not None:
                # contiguous case: frame shape (C, n_new)
                C_local, N_local = frame.shape[0], frame.shape[1]
                for k in range(N_local):
                    emg_col = [frame[ch][k] for ch in range(C_local)]
                    batch_rows.append(
                        emg_col + [imu_seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz, t_write]
                    )
            else:
                # wrap case: write part1 then part2
                # part1: (C, L-prev_idx_old)
                start1 = None  # not needed; we directly index from buffer
                # We know wrap happened from old prev_idx to end (L-1), then 0..idx-1.
                old_start = (idx - n_new) % L
                # first pass: old_start .. L-1
                k = old_start
                while k < L:
                    emg_col = [buf[ch][k] for ch in range(n_channels)]
                    batch_rows.append(
                        emg_col + [imu_seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz, t_write]
                    )
                    k += 1
                # second pass: 0 .. idx-1
                k = 0
                while k < idx:
                    emg_col = [buf[ch][k] for ch in range(n_channels)]
                    batch_rows.append(
                        emg_col + [imu_seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz, t_write]
                    )
                    k += 1

            #print(f"[DBG] n_new={n_new} contiguous={frame is not None} rows_out={len(batch_rows)} idx={idx}")
            #if frame is not None:
            #    print(f"[DBG] frame.shape={frame.shape}")

            writer.writerows(batch_rows)
            rows_since_flush += len(batch_rows)
            rows_written += len(batch_rows)
            total_rows = getattr(writer, "line_num", None)


            # periodic flush to ensure data hits disk during long runs
            if rows_since_flush >= args.flush_every:
                fout.flush()
                os.fsync(fout.fileno())
                rows_since_flush = 0

            # optional: lightweight console heartbeat once/sec
            if int(time.time() - t0) != int(t_write - t0):
                rate_rows = (t_write - t0)
                # don’t spam—comment this line out if undesired
                print(f"[LOG] rows={rows_written} (C={n_channels})  elapsed={t_write - t0:5.1f}s", end="\r")

    except KeyboardInterrupt:
        print("\n[STOP] Closing…")
    finally:
        try:
            fout.flush()
            os.fsync(fout.fileno())
            fout.close()
        except Exception:
            pass
        try:
            imu.stop()
        except Exception:
            pass
        try:
            rhx.stop_streaming()
        except Exception:
            pass
        print(f"[DONE] Saved: {os.path.abspath(args.outfile)}")


if __name__ == "__main__":
    main()
