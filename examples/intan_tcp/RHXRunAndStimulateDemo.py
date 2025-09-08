#! /bin/env python3
# Adrian Foy September 2023

"""Example demonstrating connecting to RHX software via TCP and controlling a
Stimulation/Recording Controller. This script sets up stimulation parameters
on a single channel, and runs for 5 seconds (stimulating 5 times in total).

In order to run this example script successfully, the IntanRHX software should
be started with a Stimulation/Recording Controller (or a synthetic
Stimulation/Recording Controller); other controller types will not work with
this script.

Through Network -> Remote TCP Control:

Command Output should open a connection at 127.0.0.1, Port 5000.
Status should read "Pending"

Once this port is opened, this script can be run to use TCP commands to
configure stimulation on channel A-010. Then, the controller will be run for 5
seconds, and every ~1 second a TCP command to trigger stimulation will be sent.
"""

import time
import socket


def RunAndStimulateDemo():
    """Connects via TCP to RHX software, sets up stimulation parameters on a
    single channel, and runs for 5 seconds (stimulating 5 times in total).
    """

    # Connect to TCP command server - default home IP address at port 5000
    print('Connecting to TCP command server...')
    scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scommand.connect(('127.0.0.1', 5000))

    # Query controller type from RHX software.
    # Throw an error and exit if controller type is not Stim.
    scommand.sendall(b'get type')
    commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
    isStim = commandReturn == "Return: Type ControllerStimRecord"
    if not isStim:
        raise InvalidControllerType(
            'This example script should only be used with a '
            'Stimulation/Recording Controller.'
        )

    # Query runmode from RHX software
    scommand.sendall(b'get runmode')
    commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
    isStopped = commandReturn == "Return: RunMode Stop"

    # If controller is running, stop it
    if not isStopped:
        scommand.sendall(b'set runmode stop')
        time.sleep(0.1)

    # Send commands to configure some stimulation parameters on channel A-010,
    # and execute UploadStimParameters for that channel.
    scommand.sendall(b'set a-010.stimenabled true')
    time.sleep(0.1)
    scommand.sendall(b'set a-010.source keypressf1')
    time.sleep(0.1)
    scommand.sendall(b'set a-010.firstphaseamplitudemicroamps 10')
    time.sleep(0.1)
    scommand.sendall(b'set a-010.firstphasedurationmicroseconds 500')
    time.sleep(0.1)
    scommand.sendall(b'execute uploadstimparameters a-010')
    time.sleep(1)

    # Send command to set board running
    scommand.sendall(b'set runmode run')

    # Every second for 5 seconds, execute a ManualStimTriggerPulse command
    print("Acquiring data, and stimulating every second")
    for _ in range(5):
        time.sleep(1)
        scommand.sendall(b'execute manualstimtriggerpulse f1')
    time.sleep(0.1)

    # Send command to RHX software to stop recording
    scommand.sendall(b'set runmode stop')
    time.sleep(0.1)

    # Close TCP socket
    scommand.close()


class InvalidControllerType(Exception):
    """Exception returned when received controller type is not
    ControllerStimRecord (this script only works with Stim systems).
    """


if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket
    # This is the maximum number of bytes expected for 1 read. 1024 is plenty
    # for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024

    RunAndStimulateDemo()
