"""Serial driver for the New Focus picomotor controller.

Thin wrapper over the controller's text command set used by PhysicalAlignment.py:
send() talks to the controller, setup()/cleanup() bracket a session, and move()
issues a relative move on one axis.
"""

import serial
import time

# ============================== USER SETTINGS ==============================
PORT = 'COM3'      # USB-serial adapter port
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

def send(cmd, timeout=2.0):
    ser.reset_input_buffer()
    ser.write((cmd + '\r').encode())
    buf = b''
    deadline = time.time() + timeout
    while time.time() < deadline:
        buf += ser.read(ser.in_waiting or 1)
        if buf.endswith(b'?') or buf.endswith(b'>'):
            break

    decoded = buf.decode(errors='replace').strip(' \r\n?>')
    print(f'  >> {cmd!r}  =>  {decoded!r}')
    return decoded

def setup(velocity):
    send('jof')             # joystick off
    send('def')             # default parameters (v = 2 kHz, a = 32,000 steps/s^2)
    send('typ a1 0=0')      # 0 = standard picomotor
    send('typ a1 1=0')
    send('typ a2 0=0')
    send('typ a2 1=0')
    send(f'vel a1 0={velocity}')
    send(f'vel a1 1={velocity}')
    send(f'vel a2 0={velocity}')
    send(f'vel a2 1={velocity}')
    send('mon')             # enable all drivers

def move(axis, direction, step, velocity, driver):
    send(f'chl a{driver}={axis}')
    if driver == 2:
        send(f'rel a{driver} ={-step*direction}')   # driver 2 is mounted inverted
    else:
        send(f'rel a{driver} ={step*direction}')
    send('go')
    time.sleep(abs(step) / velocity)

def get_pos():
    while True:
        response = send('pos a2')
        for line in response.splitlines():
            if '=' in line:
                return int(line.split('=')[-1])

def cleanup():
    send('jon')             # joystick back on
    send('vel a1 0=2000')
    send('vel a1 1=2000')
    send('vel a2 0=2000')
    send('vel a2 1=2000')
    send('mof')             # disable all drivers
    ser.close()
