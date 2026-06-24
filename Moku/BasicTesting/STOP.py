"""Emergency stop: halt the picomotors and restore a safe idle state.

Standalone (own serial connection) so it works even if another script left the
controller mid-move. Stops motion, re-enables the joystick, and closes the port.
"""

import serial
import time

# ============================== USER SETTINGS ==============================
PORT = 'COM3'      # USB-serial adapter port (find values in Device Manager)
BAUDRATE = 19200
# ===========================================================================

ser = serial.Serial(
    port=PORT,
    baudrate=BAUDRATE,
    bytesize=8,
    parity='N',
    stopbits=1,
    timeout=1
)

def send(cmd):
    ser.reset_input_buffer()
    ser.write((cmd + '\r').encode())
    time.sleep(0.06)
    response = b''
    while ser.in_waiting:
        response += ser.read(ser.in_waiting)
        time.sleep(0.05)
    decoded = response.decode().strip()
    print(f'  >> {cmd!r}  =>  {decoded!r}')
    return decoded


send('sto')   # stop all motion

# Restore a safe idle state
send('jon')   # joystick on
send('vel a1 0=2000')
send('vel a1 1=2000')
send('vel a2 0=2000')
send('vel a2 1=2000')
send('mof')   # disable all drivers

# If the joystick won't move the motors and all three driver lights are on,
# hold "set axis" then click the top of the joystick.
ser.close()
