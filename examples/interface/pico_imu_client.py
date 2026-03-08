import socket
import struct
import threading
import time
from collections import deque
from typing import Optional, Tuple

class PicoIMUClient:
    """
    Auto-discovery + UDP telemetry client for Pico 2 W (AP mode).
    Protocol:
      PC -> Pico:  "HELLO_PICO v1 port=<LOCAL_PORT>"  (unicast to 192.168.4.1:9000)
      Pico -> PC:  "PICO_READY v1"  (from 9000 -> LOCAL_PORT), then telemetry
      Telemetry packet (little endian): <I9f = seq(uint32), 9 floats
    """

    PACK_FMT = "<I9f"   # seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz
    PACK_SIZE = struct.calcsize(PACK_FMT)

    def __init__(
        self,
        ap_ip: str = "192.168.4.1",
        discovery_port: int = 9000,
        bufsize: int = 512,
        print_rate_hz: float = 0.0,  # set >0 to print stats periodically
    ):
        self.ap_ip = ap_ip
        self.discovery_port = discovery_port
        self.buf = deque(maxlen=bufsize)
        self._sock = None
        self._rx_th = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._print_rate_hz = print_rate_hz

        # diagnostics
        self._pkt_count = 0
        self._t_start = None
        self._last_seq = None
        self._last_from = None
        self._last_error: Optional[str] = None

    # ---------- public API ----------
    def start(self, timeout: float = 6.0) -> bool:
        """
        Start discovery + background receiver. Returns True if ready.
        """
        if self._rx_th and self._rx_th.is_alive():
            return True

        self._stop.clear()
        self._ready.clear()
        self._rx_th = threading.Thread(target=self._rx_loop, args=(timeout,), daemon=True)
        self._rx_th.start()

        # wait for discovery
        return self._ready.wait(timeout + 1.0)

    def stop(self):
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._rx_th:
            self._rx_th.join(timeout=1.0)

    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise ConnectionError(f"Failed to connect to Pico IMU: {self.last_error()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def get_latest(self) -> Optional[Tuple[int, float, float, float, float, float, float, float, float, float]]:
        """
        Return the newest telemetry tuple:
        (seq, roll, pitch, yaw, ax, ay, az, gx, gy, gz)
        """
        return self.buf[-1] if self.buf else None

    def last_seq(self) -> Optional[int]:
        return self._last_seq

    def rate_hz(self) -> float:
        if not self._t_start:
            return 0.0
        dt = time.time() - self._t_start
        return (self._pkt_count / dt) if dt > 0 else 0.0

    def last_error(self) -> Optional[str]:
        return self._last_error

    # ---------- internals ----------
    def _bind_on_ap_interface(self) -> Tuple[str, int]:
        """Bind a UDP socket to the Wi-Fi AP interface (192.168.4.x) with an ephemeral port."""
        # Find our local IP on the Pico AP interface (no packets actually sent)
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((self.ap_ip, self.discovery_port))
            local_ip = probe.getsockname()[0]
        finally:
            probe.close()

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.bind((local_ip, 0))
        self._sock = s
        return local_ip, s.getsockname()[1]

    def _discover(self, sock: socket.socket, local_ip: str, local_port: int, timeout: float) -> bool:
        """Unicast discovery to Pico AP; keep the SAME socket for reply + telemetry."""
        hello = f"HELLO_PICO v1 port={local_port}".encode()
        deadline = time.time() + timeout
        got_ready = False

        while time.time() < deadline and not self._stop.is_set():
            try:
                sock.sendto(hello, (self.ap_ip, self.discovery_port))
            except OSError as e:
                self._last_error = f"send HELLO_PICO: {e}"

            try:
                data, addr = sock.recvfrom(256)
                if data.startswith(b"PICO_READY"):
                    self._last_from = addr
                    got_ready = True
                    break
            except socket.timeout:
                pass
            except OSError as e:
                self._last_error = f"recv READY: {e}"

            time.sleep(0.2)

        return got_ready

    def _rx_loop(self, timeout: float):
        try:
            local_ip, local_port = self._bind_on_ap_interface()
            if not self._discover(self._sock, local_ip, local_port, timeout):
                self._last_error = "Discovery timeout"
                return

            # ready; allow caller to proceed
            self._t_start = time.time()
            self._pkt_count = 0
            self._ready.set()

            # receive telemetry on the SAME socket/port
            self._sock.settimeout(1.0)
            t_last_print = time.time()

            while not self._stop.is_set():
                try:
                    data, addr = self._sock.recvfrom(self.PACK_SIZE)
                except socket.timeout:
                    continue
                except OSError as e:
                    self._last_error = f"recv telem: {e}"
                    continue

                if len(data) != self.PACK_SIZE:
                    # ignore non-telemetry frames quietly
                    continue

                seq, r, p, y, ax, ay, az, gx, gy, gz = struct.unpack(self.PACK_FMT, data)
                self._last_seq = seq
                self._last_from = addr
                self._pkt_count += 1
                self.buf.append((seq, r, p, y, ax, ay, az, gx, gy, gz))

                # optional periodic stats
                if self._print_rate_hz > 0:
                    now = time.time()
                    if now - t_last_print >= 1.0 / self._print_rate_hz:
                        t_last_print = now
                        print(f"[PicoIMUClient] from {addr[0]}  rate={self.rate_hz():.1f} Hz  seq={seq}")

        finally:
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None


def main():
    """Standalone test for Pico IMU client."""
    import argparse
    
    ap = argparse.ArgumentParser(description="Pico IMU Client Test")
    ap.add_argument("--ap-ip", default="192.168.4.1", help="Pico AP IP address")
    ap.add_argument("--port", type=int, default=9000, help="Discovery port")
    ap.add_argument("--timeout", type=float, default=6.0, help="Discovery timeout (s)")
    ap.add_argument("--print-rate", type=float, default=1.0, help="Print stats every N seconds (0=off)")
    args = ap.parse_args()

    print(f"[IMU] Connecting to Pico at {args.ap_ip}:{args.port}...")
    
    try:
        with PicoIMUClient(
            ap_ip=args.ap_ip,
            discovery_port=args.port,
            print_rate_hz=args.print_rate
        ) as imu:
            print(f"[OK] Connected! Receiving at ~{imu.rate_hz():.1f} Hz")
            print("[RUN] Streaming... Press Ctrl+C to stop.\n")
            
            try:
                while True:
                    latest = imu.get_latest()
                    if latest:
                        seq, r, p, y, ax, ay, az, gx, gy, gz = latest
                        print(f"[IMU] seq={seq:5d} "
                              f"RPY=({r:+6.2f},{p:+6.2f},{y:+6.2f}) "
                              f"Acc=({ax:+5.2f},{ay:+5.2f},{az:+5.2f}) "
                              f"Gyro=({gx:+6.1f},{gy:+6.1f},{gz:+6.1f})  "
                              f"{imu.rate_hz():.1f} Hz", end='\r')
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n[STOP] Disconnecting...")
        
        print("[DONE] Connection closed.")
    
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
