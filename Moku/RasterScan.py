# This is starting with the mirror pointed towards the bottom left corner
# The idea of this code is to get training data to test the neural network
# The laser must start in the center of the iris
from moku.instruments import Oscilloscope

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

import sys
sys.path.append(r'c:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust')
import picoMotor as pico 

import time


def plot():
    plt.clf()
    plt.scatter(x_scan, y_scan, s=power)  # s= size of the point
    plt.xlabel('X-steps')
    plt.ylabel('Y-steps')
    plt.title('Laser beam power based on location')
    plt.savefig('RasterScan.png', dpi=150, bbox_inches='tight')

def contourf_plot():
    plt.close('all')                    # close all existing figures
    fig, ax = plt.subplots()

    xi = np.linspace(min(x_scan), max(x_scan), 200)
    yi = np.linspace(min(y_scan), max(y_scan), 200)   # only up to current data
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x_scan, y_scan), power, (Xi, Yi), method='linear')

    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap='hot')
    fig.colorbar(cf, ax=ax, label='Milliwatts')
    ax.set_xlabel('X-Steps')
    ax.set_ylabel('Y-Steps')
    ax.set_title('Power at Different Locations on an Iris')

    fig.savefig('Contourf.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_data():
    np.savez('scan_data_iris2.1.npz',
             x_scan=np.array(x_scan),
             y_scan=np.array(y_scan),
             power=np.array(power))
    print(f'Saved {len(power)} data points to scan_data_iris2.1.npz')

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# Setup
pico.setup(100)         

# Edit these for scan size
length = 1200
height = 1200
lines = 30 # Higher number for finer data
Lfactor = 1.18 # Increase this if scan shifts left (Amount of left steps for each right step)
Ufactor = .85  # Increase if the plot isn't elliptical enough (Amount of up steps for each right step)
Dfactor = 1.25  # Increase this if the plot starts too low (Amount of down steps for each right step)
iris = 2

if iris == 2:
    length = -length

y_steps = 0
pos = 0
x_scan = []
y_scan = []
power = []

plt.ion()

# Get the laser to the corner
pico.send(f'chl a{iris}=1')
pico.send(f'rel a{iris} ={(int)((Dfactor*height)/2)}')
pico.send('go')
time.sleep(((Dfactor*height)/2)/95)

pico.send(f'chl a{iris}=0')
pico.send(f'rel a{iris} ={(int)((-Lfactor*length)/1.75)}')
pico.send('go')
time.sleep(((Lfactor*abs(length))/1.75)/95)

try:

    osc.set_timebase(0, 0.01) # Define the length of data you get with respect to time (+- .001 seconds from trigger point)

    while True: # Continuously read and print data from input 1

        pos = 0
        pico.send(f'chl a{iris}=0')
        pico.send(f'rel a{iris} ={length}')
        pico.send('go')

        while (pos < abs(length)):
            pos = -pico.get_pos()
            #print(pos)

            data = osc.get_data()
            avg = sum(data['ch1'])/len(data['ch1'])

            power.append(avg*6.45) # Analog input x6.455 for Watts
            x_scan.append(pos)
            y_scan.append(y_steps)

        if len(np.unique(y_scan)) > 1: # Countour can't plot with one line of data
            contourf_plot()

        # Return to start
        pico.send(f'vel a{iris} 0=1000')
        pico.send(f'rel a{iris} ={(int)(-length*Lfactor)}')
        pico.send('go')
        time.sleep((abs(length)*Lfactor)/950) # Divide steps by a little less than velocity for correct delay
        pico.send(f'vel a{iris} 0=100')

        # Go up
        pico.send(f'chl a{iris}=1')
        pico.send(f'rel a{iris} =-{(int)((Ufactor*height)/lines)}')
        pico.send('go')
        time.sleep(((Ufactor*height)/lines)/95)
        y_steps += height/lines

        time.sleep(.5) #Let the motor cool down so there isn't as much drift

        if y_steps > height:
            pico.cleanup()
            osc.relinquish_ownership()
            save_data()
            break

except KeyboardInterrupt: #Ctrl C to end
    print('Scan interrupted')
    osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time
    pico.cleanup()