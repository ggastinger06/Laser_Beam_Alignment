import serial
import time

ser = serial.Serial(  # This sets up the serial communication
    port='COM3',      # USB-serial adapter port
    baudrate=19200,   
    bytesize=8,
    parity='N',
    stopbits=1,
    timeout=1
)

def send(cmd, timeout=2.0): # Waits for a response to move on
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
    send('jof')             # Joystick off
    send('def')             # Default Parameters (v = 2 kHz, a = 32,000 steps/s^2)
    send('typ a1 0=0')      # 0 = standard picomotor
    send('typ a1 1=0')     
    send('typ a2 0=0')      # 0 = standard picomotor
    send('typ a2 1=0')     
    send(f'vel a1 0={velocity}')    # Velocity of Motor A = 100 
    send(f'vel a1 1={velocity}')    # Velocity of Motor B
    send(f'vel a2 0={velocity}')    # Velocity of Motor A = 100 
    send(f'vel a2 1={velocity}')    # Velocity of Motor B
    send('mon')             # Enables all drivers

def move(axis, direction, step, velocity, driver):
    send(f'chl a{driver}={axis}')
    if driver == 2: send(f'rel a{driver} ={-step*direction}')
    else:           send(f'rel a{driver} ={step*direction}')
    send('go')
    time.sleep(abs(step)/(velocity*.95))

def get_pos():
    while True:
        response = send('pos a2')
        for line in response.splitlines():
            if '=' in line:
                return int(line.split('=')[-1])
            
def cleanup():
    send('jon')
    send('vel a1 0=2000')   
    send('vel a1 1=2000')  
    send('vel a2 0=2000')   
    send('vel a2 1=2000')  
    send('mof')
    ser.close()