import board
import busio
import digitalio
import time

# SPI pin definitions (Raspberry Pi Pico pins)
SCLK = board.GP2
MOSI = board.GP3
MISO = board.GP4  # Chip 1 MISO
CS = board.GP5

# Chip Select (CS) pin setup
cs = digitalio.DigitalInOut(CS)
cs.direction = digitalio.Direction.OUTPUT
cs.value = True  # Chip deselected initially

# Initialize SPI
spi = busio.SPI(clock=SCLK, MOSI=MOSI, MISO=MISO)

# Wait for SPI lock and configure
while not spi.try_lock():
    pass
spi.configure(baudrate=500000, phase=0, polarity=0)
spi.unlock()

def send_read_command(regnum):
    command = (0b11000000 << 8) | (regnum << 8)
    buf = command.to_bytes(2, 'big')
    result = bytearray(2)

    while not spi.try_lock():
        pass
    cs.value = False
    spi.write_readinto(buf, result)
    cs.value = True
    spi.unlock()

    return int.from_bytes(result, 'big')

def send_write_command(regnum, data):
    command = (0b10000000 << 8) | (regnum << 8) | data
    buf = command.to_bytes(2, 'big')

    while not spi.try_lock():
        pass
    cs.value = False
    spi.write(buf)
    cs.value = True
    spi.unlock()

def calibrate_chip():
    calibration_cmd = 0b0101010100000000
    buf = calibration_cmd.to_bytes(2, 'big')

    while not spi.try_lock():
        pass
    cs.value = False
    spi.write(buf)
    cs.value = True
    spi.unlock()

    time.sleep(0.01)

    for _ in range(9):
        send_read_command(40)

# Main test execution
print("Starting Intan SPI test (single chip)...")

try:
    calibrate_chip()
    time.sleep(0.01)  # Short delay after calibration

    reg_40_response = send_read_command(40)
    print(f"Response from Register 40: {reg_40_response}")

    send_write_command(14, 0x01)
    print("Successfully wrote to Register 14.")

except Exception as e:
    print(f"SPI communication error: {e}")

print("Test complete.")
