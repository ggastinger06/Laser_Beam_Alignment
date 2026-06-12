# This is starting with the mirror pointed towards the bottom left corner
# The idea of this code is to get training data to test the neural network
# The laser must start in the center of the iris
from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

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

def plot():
    plt.clf()
    plt.scatter(x_scan, y_scan, s=power)  # s= size of the point
    plt.xlabel('X-steps')
    plt.ylabel('Y-steps')
    plt.title('Laser beam power based on location')
    plt.pause(0.01)
    
def threeDplot():
    # Create a regular grid to interpolate onto
    xi = np.linspace(min(x_scan), max(x_scan), 100)
    yi = np.linspace(min(y_scan), max(y_scan), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate scattered power data onto the grid
    Zi = griddata((x_scan, y_scan), power, (Xi, Yi), method='linear')

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(Xi, Yi, Zi, cmap=cm.hot, vmin=np.nanmin(Zi))

    ax.set_xlabel('X-steps')
    ax.set_ylabel('Y-steps')
    ax.set_zlabel('Power')
    ax.set_title('Laser beam power based on location')
    plt.show()

    time.sleep(120)

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# Setup
send('jof')             # Joystick off
send('def')             # Default Parameters (v = 2 kHz, a = 32,000 steps/s^2)
send('typ a1 0=0')      # 0 = standard picomotor
send('vel a1 0=100')    # Velocity of Motor A = 100 kHz (Defined as "fine" in the manual")

send('typ a1 1=0')      # 0 = standard picomotor
send('vel a1 1=100')    # Velocity of Motor B

send('mon')             # Enables all drivers

length = 2500
height = 2500
y_steps = 0
pos = 0
go_right = True

x_scan = []
y_scan = []
power = []

plt.ion() 

try:
    
    osc.set_timebase(0, 0.01) # Define the length of data you get with respect to time (+- .001 seconds from trigger point)

    while True: # Continuously read and print data from input 1
        
        pos = 0
        send('chl a1=0')
        if go_right:
            # Go right
            send('rev a1')
        else:
            send('for a1')

        send('go')        
        time.sleep(.2)

        while (abs(pos) < length):
            pos = abs(get_pos())
            if go_right: # It's currently going right
                x_scan.append(pos)
            else:
                x_scan.append(length - abs(pos))
            print(pos)
            
            data = osc.get_data()
            avg = sum(data['ch1'])/len(data['ch1'])
            power.append(avg*50)

            y_scan.append(y_steps)
        
        # Go up
        send('sto')
        send('chl a1=1') 
        send('rel a1 =-100')
        send('go a1')
        time.sleep(1)
        send('sto')
        y_steps += 100
        y_scan.append(y_steps)

        # Make sure to update the rest of the data
        if go_right:
            x_scan.append(2000)
            go_right = False
        else:
            x_scan.append(0)
            go_right = True
        data = osc.get_data()
        avg = sum(data['ch1'])/len(data['ch1'])
        power.append(avg*50)

        plot()
        # Plot the graph everytime you move up

        if y_steps >= height:
            cleanup()

except KeyboardInterrupt: #Ctrl C to end
    cleanup()