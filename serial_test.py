import serial
import time

# Configure the serial connection
COM_PORT = 'COM5'  # Known COM port for Pico
BAUD_RATE = 9600    # Baud rate set to match Pico's configuration

# Initialize the serial connection to the Pico
with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
    # Allow a moment for the connection to stabilize
    time.sleep(1)
    
    # Send a command, for example, 'flex' to the Pico
    command = 'extend;'  # Adding ';' as a terminator (based on your setup)
    ser.write(command.encode('utf-8'))
    print(f"Sent command: {command.strip()}")

    # Wait for and read any response from the Pico
    while True:
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Received response from Pico: {response}")
        else:
            print("No response from Pico")
            break

# The serial connection will close automatically after exiting the 'with' block
