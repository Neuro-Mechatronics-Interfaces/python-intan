"""
  (c) Jonathan Shulgach - Cite and Notice license:
    All modifications to this code or use of it must include this notice and give credit for use.
    Credit requirements:
      All publications using this code must cite all contributors to this code.
      A list must be updated below indicating the contributors alongside the original or modified code appropriately.
      All code built on this code must retain this notice. All projects incorporating this code must retain this license text alongside the original or modified code.
      All projects incorporating this code must retain the existing citation and license text in each code file and modify it to include all contributors.
      Web, video, or other presentation materials must give credit for the contributors to this code, if it contributes to the subject presented.
      All modifications to this code or other associated documentation must retain this notice or a modified version which may only involve updating the contributor list.
    
    Primary Authors:
      - Jonathan Shulgach, PhD Student - Neuromechatronics Lab, Carnegie Mellon University
      
    Corresponding copyright notices:
      - SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
      - SPDX-License-Identifier: MIT

   Other than the above, this code may be used for any purpose and no financial or other compensation is required.
   Contributors do not relinquish their copyright(s) to this software by virtue of offering this license.
   Any modifications to the license require permission of the authors.
   
   Description:
      This Python code controls a 3D printed robot arm. it uses a Raspberry Pico 2 as the main microcontroller, a PWM driver
      and servos for controlling the wrist and fingers. It uses serial parsing for command inputs and outputs
"""
import time
import busio
import board
import asyncio
from adafruit_servokit import ServoKit
from usbserialreader import USBSerialReader

__author__ = "Jonathan Shulgach"
__version__ = "0.0.1"


class Arm(object):
    """ Object for controlling the 3D printed robot arm usig Circuitpython
    
        Parameters:
        -----------
        name                : (str)   The custom name for the class
        simulate            : (bool)  Allow physical control of hardware
        use_uart            : (bool)  Set to false to use the console, True for UART pins
        command_delimiter   : (str)   Multiple commands should be parsed too
        argument_delimiter  : (str)   Expecting arguements to be spaced
        verbose             : (bool)  Enable/disable verbose output text to the terminal
    """
    def __init__(self, name='HumanArm',
                       simulate=False,
                       use_uart=False,
                       command_delimiter=";",
                       argument_delimiter=":",
                       verbose=False,
                       ):
        self.name = name
        self.simulate = simulate
        self.use_uart=use_uart
        self.command_delimiter=command_delimiter
        self.argument_delimiter=argument_delimiter
        self.verbose=verbose
        self.all_stop = False
        self.connected = False
        self.usb_serial = USBSerialReader(use_UART=self.use_uart, TERMINATOR=self.command_delimiter, verbose=self.verbose)
        self.servos = None
        if not self.simulate:
            # necessary to create the custom busio.I2C object instead of the ServoKit default due to some dormant library issue 
            i2c = busio.I2C(board.GP1, board.GP0)
            self.servos = ServoKit(channels=16, i2c=i2c) 
        self.logger("USB Serial Parser set up. Reading serial commands")
        
    def logger(self, *argv, warning=False):
        """ Robust printing function """
        msg = ''.join(argv)
        if warning: msg = '(Warning) ' + msg
        print("[{:.3f}][{}] {}".format(time.monotonic(), self.name, msg))
        
    def start(self):
        """ Makes a call to the asyncronous library to run a main routine """
        asyncio.run(self.main())  # Need to pass the async function into the run method to start

    def stop(self):
        """ Sets a flag to stop running all tasks """
        self.all_stop = True
        
    async def main(self):
        """ Start main tasks and coroutines in a single main function """        
        asyncio.create_task(self.serial_client())
        while self.all_stop != True:
            await asyncio.sleep(0) # Calling #async with sleep for 0 seconds allows coroutines to run
            
    async def serial_client(self, interval=100):
        """ Read serial commands and add them to the command queue """
        while self.all_stop != True:
            self.usb_serial.update()
            if self.usb_serial._out_data:        
                await self.parse_command(self.usb_serial.out_data) # Handle serial commands as soon as they come
            await asyncio.sleep(1 / int(interval))
            
    async def parse_command(self, msg_list):
        """ Parses serial messages to control joint positions """
        print(f"Received {msg_list}")
        for msg in msg_list:
            cmd=msg[0]
            if cmd == 'flex':
                servo_positions = {'Wrist':30, 'Thumb':30, 'Index':30, 'Ring':30, 'Middle':30, 'Pinky':30}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)
            elif cmd == 'extend':
                servo_positions = {'Wrist':30, 'Thumb':30, 'Index':30, 'Ring':30, 'Middle':30, 'Pinky':30}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)
            elif cmd == 'thumb':
                self.update_servo('Thumb', 90)
            elif cmd == 'index':
                self.update_servo('Index', 90)
            elif cmd == 'ring':
                self.update_servo('Ring', 90)
            elif cmd == 'middle':
                self.update_servo('Middle', 90)
            elif cmd == 'pinky':
                self.update_servo('Pinky', 90)
            elif cmd == 'open':
                self.update_servo('Wrist', 0)
            elif cmd == 'grip':
                self.update_servo('Wrist', 180)
            elif cmd == 'pinch':
                self.update_servo('Thumb', 45)
                self.update_servo('Index', 45)
            elif cmd == 'rest':
                servo_positions = {'Wrist':0, 'Thumb':0, 'Index':0, 'Ring':0, 'Middle':0, 'Pinky':0}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)
            else:
                self.logger(f"Unknown command received: '{cmd}'", warning=True)
        
    def update_servo(self, key, servo_val):
        """ Update the phsyical servo configured from the ServoKit, given specific index and angle value """
        try:
            self.logger(f"{key} set to {servo_val} degrees")
            if not self.simulate:
                self.servos.servo[key].angle = servo_val 
        except ValueError as e:
            print(f"Error updating servo {key}: {e}")


if __name__ == "__main__":
    
    arm = Arm('3D-printed Arm', simulate=False, use_uart=False, verbose=True)
    try:
        arm.start()
    except KeyboardInterrupt:
        arm.stop()
    