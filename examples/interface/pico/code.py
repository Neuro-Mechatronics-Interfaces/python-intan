# code.py — CircuitPython 9.x, Pico 2 W (AP + single-socket discovery/telemetry with robust send)
import wifi
import time
import math
import board
import busio
import struct
import socketpool
from micropython import const

# Using built-in BNO085
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ROTATION_VECTOR,
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_GYROSCOPE
)

SSID = "PICO2W_AP"
PASSWORD = "pico-pass"

DISCOVERY_PORT = const(9000)
RATE_HZ = const(20)           # test at 20 Hz first; bump up after it works
PACK_FMT = "<I9f"
PACK_SIZE = struct.calcsize(PACK_FMT)

BNO08X_ADDRESS = 0x4B  # or 0x4A depending on wiring

TARGET_HZ = 100

report_us = int(1_000_000 / TARGET_HZ)

class PicoUDPStreamer:
    def __init__(self):
        self.seq = 0
        self.period = 1.0 / RATE_HZ
        self.buf = bytearray(PACK_SIZE)
        self.mv = memoryview(self.buf)  # <-- immutable view for sendto
        self.pool = None
        self.sock = None
        self.dst = None
        self.yaw_offset = None

        self.i2c = busio.I2C(board.GP5, board.GP4, frequency=400_000)  # or 100k if flaky
        self.bno = BNO08X_I2C(self.i2c, address=BNO08X_ADDRESS)  # 0x4A/0x4B depending on wiring
        time.sleep(0.3)
        self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR, report_interval=report_us)  # ~20 Hz
        self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION, report_interval=report_us)
        #self.bno.enable_feature(BNO_REPORT_GYROSCOPE, report_interval=report_us) # Don't need, save bandwidth
        time.sleep(0.2)



    def start_ap(self):
        print("Starting AP…")
        wifi.radio.start_ap(SSID, PASSWORD)
        time.sleep(2.0)
        print("AP active:", wifi.radio.ipv4_address_ap)
        self.pool = socketpool.SocketPool(wifi.radio)

        s = self.pool.socket(self.pool.AF_INET, self.pool.SOCK_DGRAM)
        s.bind(("0.0.0.0", DISCOVERY_PORT))
        s.settimeout(0)  # non-blocking
        self.sock = s

    def wait_for_client(self):
        rb = bytearray(128)
        print(f"Waiting for client handshake on UDP {DISCOVERY_PORT}…")
        while True:
            try:
                nbytes, addr = self.sock.recvfrom_into(rb)
            except OSError:
                time.sleep(0.02)
                continue

            msg = bytes(memoryview(rb)[:nbytes]).decode("utf-8", "ignore").strip()
            if not msg.startswith("HELLO_PICO"):
                continue

            # parse port=####
            port = 5555
            for part in msg.split():
                if part.startswith("port="):
                    try:
                        port = int(part.split("=", 1)[1])
                    except Exception:
                        pass

            self.dst = (addr[0], port)
            self.seq = 0

            # reply from same socket/port
            try:
                self.sock.sendto(b"PICO_READY v1", addr)
            except OSError as e:
                print("reply send error:", e)

            print("Client discovered:", self.dst)

            # --- send ONE immediate test frame so you should see it in Wireshark immediately
            r,p,y,ax,ay,az,gx,gy,gz = self.fake_imu()
            struct.pack_into(PACK_FMT, self.buf, 0, self.seq, r,p,y, ax,ay,az, gx,gy,gz)
            try:
                sent = self.sock.sendto(self.mv, self.dst)
                print(f"test tx bytes={sent} → {self.dst}")
                self.seq += 1
            except OSError as e:
                print("test tx error:", e)
            return

    def fake_imu(self):
        """ Dummy values for testing"""
        t = time.monotonic()
        return (0.1*t, 0.2*t, 0.3*t, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0)

    @staticmethod
    def _parse_quat(q):
        if hasattr(q, "i") and hasattr(q, "j") and hasattr(q, "k") and hasattr(q, "real"):
            return (q.real, q.i, q.j, q.k)
        if isinstance(q, (tuple, list)) and len(q) >= 4:
            x, y, z, w = q[0], q[1], q[2], q[3]
            return (w, x, y, z)
        raise ValueError("Unexpected quaternion format: {!r}".format(q))

    @staticmethod
    def _quat_to_rpy_deg(w, x, y, z):
        sinr_cosp = 2.0 * (w * x + y * z);
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))  # clamp
        pitch = math.degrees(math.asin(sinp))

        siny_cosp = 2.0 * (w * z + x * y);
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        # Normalize yaw to [-180, 180)
        yaw = ((yaw + 180.0) % 360.0) - 180.0
        return roll, pitch, yaw

    def read_imu(self):
        # returns roll, pitch, yaw (deg) + zeros for accel/gyro if you don't use them yet
        q = self.bno.quaternion  # (i, j, k, real) or similar
        w, x, y, z = self._parse_quat(q)
        pitch, roll, yaw = self._quat_to_rpy_deg(w, x, y, z)

        # Capture baseline yaw and zero subsequent readings
        if self.yaw_offset is None:
            self.yaw_offset = yaw
        yaw -= self.yaw_offset
        yaw = ((yaw + 180.0) % 360.0) - 180.0# Normalize again to [-180,180)

        # Prefer gravity-free linear acceleration (m/s^2)
        ax, ay, az = self.bno.linear_acceleration if hasattr(self.bno, 'linear_acceleration') else (0.0, 0.0, 9.8) # m/s^2, gravity removed

        # Gyro (typically rad/s in Adafruit driver)
        #gx, gy, gz = self.bno.gyro if hasattr(self.bno, "gyro") else (0.0, 0.0, 0.0)
        gx, gy, gz = (0.0, 0.0, 0.0)  # skip gyro for now to save bandwidth
        return roll, pitch, yaw, ax, ay, az, gx, gy, gz

    def run(self):
        self.start_ap()
        while True:
            self.wait_for_client()
            next_t = time.monotonic()
            rb = bytearray(128)

            while True:
                now = time.monotonic()
                if now >= next_t:
                    while next_t <= now:
                        next_t += self.period
                    #r,p,y,ax,ay,az,gx,gy,gz = self.fake_imu()
                    r,p,y,ax,ay,az,gx,gy,gz = self.read_imu()
                    struct.pack_into(PACK_FMT, self.buf, 0, self.seq, r,p,y, ax,ay,az, gx,gy,gz)
                    try:
                        sent = self.sock.sendto(self.mv, self.dst)
                        if (self.seq % 20) == 0:
                            print(f"tx bytes={sent} seq={self.seq} → {self.dst}")
                        self.seq += 1
                    except OSError as e:
                        print("tx error:", e)
                        break  # rediscover

                # accept new HELLO to hot-switch while streaming
                try:
                    nbytes, addr = self.sock.recvfrom_into(rb)
                    if nbytes:
                        msg = bytes(memoryview(rb)[:nbytes]).decode("utf-8", "ignore").strip()
                        if msg.startswith("HELLO_PICO"):
                            port = self.dst[1] if self.dst else 5555
                            for part in msg.split():
                                if part.startswith("port="):
                                    try: port = int(part.split("=", 1)[1])
                                    except Exception: pass
                            self.dst = (addr[0], port)
                            try: self.sock.sendto(b"PICO_READY v1", addr)
                            except OSError as e: print("re-HELLO reply error:", e)
                            print("Client re-discovered:", self.dst)
                except OSError:
                    pass

                time.sleep(0)

try:
    PicoUDPStreamer().run()
except Exception as e:
    print("Fatal:", e)
    raise
