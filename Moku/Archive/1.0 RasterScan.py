# This is starting with the mirror pointed towards the bottom left corner
# The idea of this code is to get training data to test the neural network
# The laser must start in the center of the iris
from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt

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

def send(cmd):
    ser.reset_input_buffer()
    ser.write((cmd + '\r').encode())
    time.sleep(0.1)
    response = b''
    while ser.in_waiting:
        response += ser.read(ser.in_waiting)
    decoded = response.decode().strip()
    print(f'  >> {cmd!r}  =>  {decoded!r}')
    return decoded

current_channel = None # Save time by not sending the change channel command is already on the channel

def move(axis, steps): # 0 = x-axis, 1 = y-axis
    global current_channel 
    if axis != current_channel:
        send(f'chl a1={axis}')
        current_channel = axis
    send(f'rel a1 ={-steps}')
    send('go a1')

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# Setup
send('jof')             # Joystick off
send('def')             # Default Parameters (v = 2 kHz, a = 32,000 steps/s^2)
send('typ a1 0=0')      # 0 = standard picomotor
send('vel a1 0=100')    # Velocity of Motor A = 250 kHz (Defined as "fine" in the manual")

send('typ a1 1=0')      # 0 = standard picomotor
send('vel a1 1=100')    # Velocity of Motor B

send('mon')             # Enables all drivers

send('chl a1=1') 
send('abs a1 500')   
send('go a1')
time.sleep(15)

send('chl a1=0') 
send('abs a1 500')
send('go a1')
time.sleep(15)

xPos = 0
yPos = 0
step = 100 #Must be equal to velocity
length = 5000
height = 5000

x_scan = []
y_scan = []
power = []

try:
    
    osc.set_timebase(0, 0.01) # Define the length of data you get with respect to time (+- .001 seconds from trigger point)

    while True: # Continuously read and print data from input 1
        data = osc.get_data()
        avg = sum(data['ch1'])/len(data['ch1'])
       
        move(0, step)
        xPos += step

        if (yPos >= height): # Done
            send('jon')
            send('vel a1 0=2000')   
            send('vel a1 1=2000')  
            send('MOF A1')
            ser.close()
            osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time
            break

        x_scan.append(xPos)
        y_scan.append(yPos)
        power.append(avg*250)


        # If edge of the frame is reached flip direction
        if  (xPos >= length) or (xPos <= 0):
            move(1, abs(step))
            yPos += abs(step)
            step = -step
            plt.clf()
            plt.scatter(x_scan, y_scan, s=power)  # s= size of the point
            plt.xlabel('X-distance')
            plt.ylabel('Y-distance')
            plt.title('Laser beam power based on location')
            plt.pause(0.01)

except KeyboardInterrupt: #Ctrl C to end
    send('jon')
    send('vel a1 0=2000')   
    send('vel a1 1=2000')  
    send('MOF A1')
    ser.close()
    osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time