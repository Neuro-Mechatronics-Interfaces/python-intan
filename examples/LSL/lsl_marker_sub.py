#!/usr/bin/env python3
import argparse
from intan.interface import LSLSubscriber

def main():
    ap = argparse.ArgumentParser(description="Quick LSL listener (Markers).")
    ap.add_argument("--type", default="Markers")
    ap.add_argument("--name", default=None)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    def on_marker(value: str, ts: float):
        print(f"[marker] {value} at {ts:.6f}")

    with LSLSubscriber(stream_type=args.type, name=args.name,
                       timeout=args.timeout, verbose=args.verbose) as sub:
        # background callback
        sub.set_callback(on_marker, poll_hz=50)
        print("Listening… Ctrl+C to quit.")
        try:
            while True:
                # you can also poll explicitly:
                # msg = sub.pull(timeout=0.0)
                # if msg: print("polled:", msg)
                pass
        except KeyboardInterrupt:
            print("\nDone.")

if __name__ == "__main__":
    main()
