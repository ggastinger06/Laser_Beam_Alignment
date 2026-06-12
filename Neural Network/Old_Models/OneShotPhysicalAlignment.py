# This is starting with the mirror pointed towards the bottom left corner
# The idea of this code is to get training data to test the neural network
# The laser must start in the center of the iris
from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

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
    #print(f'  >> {cmd!r}  =>  {decoded!r}')
    #print(decoded)
    return decoded

def get_pos():
    while True:
        response = send('pos')
        for line in response.splitlines():
            if '=' in line:
                return int(line.split('=')[-1])
        time.sleep(0.05)  # got only the prompt, retry

def cleanup():
    send('jon')
    send('vel a1 0=2000')   
    send('vel a1 1=2000')  
    send('MOF A1')
    ser.close()
    osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time

def move_get_points(axis, dir, steps): # 0=x, 1=y, 1=right/up, -1=left/down  
    instances = steps/100
    values = []
    for i in range(instances):
        send(f'chl a1={axis}') # axis
        send(f'rel a1 ={-100*dir}') # Move the laser
        send('go')
        time.sleep(.7) # 100 step chunks

        send('sto')
        time.sleep(.2)
        data = osc.get_data()
        point = sum(data['ch1'])/len(data['ch1'])

        values.append(point)

        i += 1

    send(f'REL A1 ={steps*dir}') # return to origin 
    send('go')
    time.sleep(6)
    send('sto')
    time.sleep(.2)

    return values


osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# Setup
send('jof')             # Joystick off
send('def')             # Default Parameters (v = 2 kHz, a = 32,000 steps/s^2)
send('typ a1 0=0')      # 0 = standard picomotor
send('vel a1 0=100')    # Velocity of Motor A = 100 kHz (Defined as "fine" in the manual")

send('typ a1 1=0')      # 0 = standard picomotor
send('vel a1 1=100')    # Velocity of Motor B

send('mon')             # Enables all drivers


pos = 0

plt.ion() 

try:
    
    osc.set_timebase(0, 0.01) # Define the length of data you get with respect to time (+- .001 seconds from trigger point)

    while True: # Continuously read and print data from input 1
        
        pos = 0
        data = osc.get_data()
        origin = sum(data['ch1'])/len(data['ch1'])
        time.sleep(.2)

        pxn = move_get_points(0, -1, 500)
        pxp = move_get_points(0, 1, 500)
        pyn = move_get_points(1, -1, 500)
        pyp = move_get_points(1, 1, 500)

        # derivatives between each point
        dxn = np.diff(pxn)
        dxp = np.diff(pxp)
        dyn = np.diff(pyn)
        dyp = np.diff(pyp)

        # average derivative of each
        xn = np.mean(dxn)
        xp = np.mean(dxp)
        yn = np.mean(dyn)
        yp = np.mean(dyp)

        print(f'origin = {origin}, xn = {xn}, xp = {xp}, yn = {yn}, yp = {yp}')


        # Plot
        plt.clf()

        vectors = [
            ( xp,   0),   # right
            (-xn,   0),   # left
            (  0,  yp),   # up
            (  0, -yn),   # down
        ]

        for dx, dy in vectors:
            magnitude = dx + dy   # positive = green, negative = red
            color = 'green' if magnitude > 0 else 'red'

            plt.quiver(0, 0, dx, dy,
                       angles='xy', scale_units='xy', scale=1,
                       color=color, headwidth=3, headlength=4)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(0, color='gray', linewidth=0.3)
        plt.axvline(0, color='gray', linewidth=0.3)
        plt.grid(True)
        plt.title(f'Directional Derivatives')

        plt.savefig('vectors.png', dpi=150, bbox_inches='tight')  # save before show
        plt.pause(0.1)

        time.sleep(30)   # wait before next measurement cycle

        cleanup()
        break

except KeyboardInterrupt: #Ctrl C to end
    cleanup()