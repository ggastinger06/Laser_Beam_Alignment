import serial
import time

# Found if you find the device in devices manager and type in all of the values
ser = serial.Serial(  # This sets up the serial communication
    port='COM3',      # USB-serial adapter port
    baudrate=19200,   
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

 
send('sto')

# Cleanup
send('jon')
send('vel a1 0=2000')   
send('vel a1 1=2000')   
send('vel a2 0=2000')   
send('vel a2 1=2000') 
send('mof') 

# If joystick doesn't move the motors and all three driver lights are on, hold "set axis" and then click the top of the joystick

ser.close()