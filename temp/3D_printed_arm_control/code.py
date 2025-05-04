<<<<<<< Updated upstream:3D_printed_arm_control/code.py

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
__version__ = "0.0.2"


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
        
        # Check whether to flip servo directions: wrist, thumb, index, middle, ring, pinky
        self.flip_direction = {'wrist':False, 'thumb':True, 'index':True, 'middle':False, 'ring':False, 'pinky':True} 
        self.servo_index = {'wrist':15, 'thumb':14, 'index':13, 'middle':12, 'ring':11, 'pinky':10}
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
        if self.verbose: 
            print(f"Received {msg_list}")
        for msg in msg_list:
            if not msg:
                if self.verbose: 
                    self.logger("Empty message")
                continue
                
            cmd=msg.pop(0)
            if cmd == 'set_joint':
                if len(msg) < 2:
                    if self.verbose: 
                        self.logger("set command detected, need to pass a servo key and value", warning=True)
                    continue
                    
                servo_key = msg.pop(0)
                #self.logger(f"Second argument: {servo_key}")
                
                servo_val = msg.pop(0)
                #self.logger(f"Third argument: {servo_val}")
                
                # Convert value to integer safely
                try:
                    servo_val = int(servo_val)
                except ValueError:
                    if self.verbose: 
                        self.logger(f"Invalid servo value: {servo_val}", warning=True)
                    return

                # Log and send command to update servo
                if self.verbose: 
                    self.logger(f"Setting {servo_key} to {servo_val}")
                self.update_servo(servo_key, servo_val)
            
            elif cmd == 'set_joints':
                if self.verbose: 
                    print("Received set_joints command")
                if len(msg) > 1:
                    val = int(msg.pop(0))
                    #self.logger(f"Setting wrist to {val}")
                    self.update_servo('wrist', val)
                    if len(msg) > 0:
                        val = int(msg.pop(0))
                        #self.logger(f"Setting thumb to {val}")
                        self.update_servo('thumb', val)
                        if len(msg) > 0:
                            val = int(msg.pop(0))
                            #self.logger(f"Setting index to {val}")
                            self.update_servo('index', val)
                            if len(msg) > 0:
                                val = int(msg.pop(0))
                                #self.logger(f"Setting middle to {val}")
                                self.update_servo('middle', val)
                                if len(msg) > 0:
                                    val = int(msg.pop(0))
                                    #self.logger(f"Setting ring to {val}")
                                    self.update_servo('ring', val)
                                    if len(msg) > 0:
                                        val = int(msg.pop(0))
                                        #self.logger(f"Setting pinky to {val}")
                                        self.update_servo('pinky', val)                       
                
            elif cmd == 'close':
                servo_positions = {'wrist':90, 'thumb':0, 'index':120, 'ring':120, 'middle':120, 'pinky':120}
                for key, angle in servo_positions.items():
                    #time.sleep(0.2)
                    self.update_servo(key, angle)
            elif cmd == 'extend':
                servo_positions = {'wrist':90, 'thumb':0, 'index':0, 'ring':0, 'middle':0, 'pinky':0}
                for key, angle in servo_positions.items():
                    #time.sleep(0.2)
                    self.update_servo(key, angle)
            elif cmd == 'pronate':
                self.update_servo('wrist', 0)
            elif cmd == 'supinate':
                self.update_servo('wrist', 180)                
            elif cmd == 'thumb':
                self.update_servo('thumb', 180)
            elif cmd == 'index':
                self.update_servo('index', 120)
            elif cmd == 'ring':
                self.update_servo('ring', 120)
            elif cmd == 'middle':
                self.update_servo('middle', 120)
            elif cmd == 'pinky':
                self.update_servo('pinky', 120)
            elif cmd == 'open':
                servo_positions = {'wrist':90, 'thumb':0, 'index':0, 'ring':0, 'middle':0, 'pinky':0}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)
            elif cmd == 'grip':
                servo_positions = {'wrist':90, 'thumb':120, 'index':120, 'ring':120, 'middle':120, 'pinky':120}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)
            elif cmd == 'pinch':
                self.update_servo('thumb', 180)
                self.update_servo('index', 180)
            elif cmd == 'point':
                servo_positions = {'wrist':90, 'thumb':180, 'index':0, 'ring':180, 'middle':180, 'pinky':180}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)                
            elif cmd == 'spiderman':
                servo_positions = {'wrist':90, 'thumb':0, 'index':0, 'ring':180, 'middle':180, 'pinky':0}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)                
            elif cmd == 'rest':
                servo_positions = {'wrist':90, 'thumb':0, 'index':0, 'ring':0, 'middle':0, 'pinky':0}
                for key, angle in servo_positions.items():
                    self.update_servo(key, angle)
            else:
                self.logger(f"Unknown command received: '{cmd}'", warning=True)
        
    def update_servo(self, key, servo_val):
        """ Update the phsyical servo configured from the ServoKit, given specific index and angle value """
        try:
            if self.verbose: 
                self.logger(f"{key} set to {servo_val} degrees")
            if not self.simulate:
                if self.flip_direction[key]: servo_val = 180-servo_val
                idx = self.servo_index[key]
                self.servos.servo[idx].angle = servo_val 
        except ValueError as e:
           if self.verbose: 
               print(f"Error updating servo {key}: {e}")


if __name__ == "__main__":
    
    
    # Servo positions at rest
    #    thumb: 180
    #    index: 180
    #    middle: 0
    #    ring:  0
    #    pinky: 180
    
    
    arm = Arm('3D-printed Arm', simulate=False, use_uart=False, verbose=False)
    try:
        arm.start()
    except KeyboardInterrupt:
        arm.stop()