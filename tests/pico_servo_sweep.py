"""
Simple sweep code for testing 
"""
import time
import busio
import board
from adafruit_servokit import ServoKit


# necessary to create the custom busio.I2C object instead of the ServoKit default due to some dormant library issue 
i2c = busio.I2C(board.GP1, board.GP0)
servos = ServoKit(channels=16, i2c=i2c) 

SERVO_NUM = 15

while True:
    time.sleep(2)
    for i in range(0, 90, 1):
        servos.servo[SERVO_NUM].angle = i
        time.sleep(0.01)
        print(i)        
    time.sleep(1)
    for i in range(90, 180, 1):
        servos.servo[SERVO_NUM].angle = i
        time.sleep(0.01)
        print(i)

    time.sleep(2)
    for i in range(180, 90, -1):
        servos.servo[SERVO_NUM].angle = i 
        time.sleep(0.01)
        print(i)
    time.sleep(1)
    for i in range(90, 0, -1):
        servos.servo[SERVO_NUM].angle = i 
        time.sleep(0.01)
        print(i)
