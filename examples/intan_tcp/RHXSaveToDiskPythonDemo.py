#! /bin/env python3
# Adrian Foy September 2023

"""Example demonstrating connecting to RHX software via TCP, providing a file
location and name to save data to, and recording 5 seconds of data.

In order to run this example script successfully, the IntanRHX software
should first be started, and through Network -> Remote TCP Control:

Command Output should open a connection at 127.0.0.1, Port 5000.
Status should read "Pending".

Once this port is opened, this script can be run to ask the user for a location
to save data to, then use TCP commands to configure this save location for the
RHX software. Then, ~5 seconds of data are recorded to this save location
before stopping recording.
"""

import time
import socket
import os

import tkinter as tk
from tkinter import filedialog


def SaveToDiskDemo():
    """Connects via TCP to RHX software, communicates the file save name and
    location, and records for 5 seconds.
    """
    # Connect to TCP command server - default home IP address at port 5000
    print('Connecting to TCP command server...')
    scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scommand.connect(('127.0.0.1', 5000))

    # Query runmode from RHX software
    scommand.sendall(b'get runmode')
    commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
    isStopped = commandReturn == "Return: RunMode Stop"

    # If controller is running, stop it
    if not isStopped:
        scommand.sendall(b'set runmode stop')
        # Allow time for RHX software to accept this command before next.
        time.sleep(0.1)

    # Query controller type from RHX software. If Stim, set isStim to true
    scommand.sendall(b'get type')
    commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
    isStim = commandReturn == "Return: Type ControllerStimRecord"

    # Get file save location from user, with the correct file suffix depending
    # on controller type
    print("Please provide a save location and name for incoming data.")
    root = tk.Tk()
    root.withdraw()
    fileSuffix = ".rhs" if isStim else ".rhd"
    fullFileName = filedialog.asksaveasfilename(defaultextension=fileSuffix)
    if not fullFileName:
        print("Canceled")
        return

    # Extract the path and the basefilename (no suffix)
    path = os.path.dirname(fullFileName)
    baseFileName = os.path.basename(fullFileName)[:-4]

    # Send command to RHX software to set baseFileName
    scommand.sendall(b'set filename.basefilename '
                     + baseFileName.encode('utf-8'))
    time.sleep(0.1)

    # Send command to RHX software to set path
    scommand.sendall(b'set filename.path ' + path.encode('utf-8'))
    time.sleep(0.1)

    # Send command to RHX software to begin recording
    scommand.sendall(b'set runmode record')

    # Wait for 5 seconds
    print("Acquiring data...")
    time.sleep(5)

    # Send command to RHX software to stop recording
    scommand.sendall(b'set runmode stop')
    time.sleep(0.1)

    # Close TCP socket
    scommand.close()

    # Notify that writing to disk has been completed
    print("Data has been saved to the location: " + fullFileName)


if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket
    # This is the maximum number of bytes expected for 1 read.
    # 1024 is plenty for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024

    SaveToDiskDemo()
