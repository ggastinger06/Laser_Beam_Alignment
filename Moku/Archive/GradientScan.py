from moku.instruments import Oscilloscope
from enum import Enum

import serial
import time

ser = serial.Serial(  # This sets up the serial communication
    port='COM4',      # USB-serial adapter port
    baudrate=19200,   # Found in manual (Uses ASCII inputs)
    bytesize=8,
    parity='N',
    stopbits=1,
    timeout=1
)

def send(cmd): # Function for sending ASCII commands to the picomotors
    ser.write((cmd + '\r\n').encode())
    time.sleep(0.01)
    return ser.read_all().decode().strip()

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

"""
def focusCheck():
    send('CHL A1=1') # y axis
    send('REL A1 =-400 G') # Move the laser DOWN
    osc.get_data()
    ch1 = data['ch1']
    avgUp = sum(ch1)/len(ch1)

    send('REL A1 =800 G') # Move the laser UP
    osc.get_data()
    ch1 = data['ch1']
    avgDown = sum(ch1)/len(ch1)
    send('REL A1 =-400 G') # Move the laser DOWN

    send('CHL A1=0') # x axis
    send('REL A1 =-400 G') # Move the laser Left
    osc.get_data()
    ch1 = data['ch1']
    avgDown = sum(ch1)/len(ch1)

    send('REL A1 =800 G') # Move the laser Right
    osc.get_data()
    ch1 = data['ch1']
    avgDown = sum(ch1)/len(ch1)
    send('REL A1 =-400 G') # Move the laser Left
"""

# Initialize motors: A=0, B=1, C=2 
send('JOF')             # Disable Joystick
send('MON A1')          # Enable motor channel

# Setup motion for each motor
send('VEL A1 0=1000')   # Set velocity of motor
send('ACC A1 0=2000')   # Set acceleration of motor
send('VEL A1 1=1000')   
send('ACC A1 1=2000') 

send('CHL A1=0') # Start on x axis motor

# Make a class for directions
class Direction(Enum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3

dir = Direction.Up

last_avg = 0

try:
    
    osc.set_timebase(-0.01, 0.01) # Define the length of data you get with respect to time (+- .01 seconds from trigger point)

    while True: # Continuously read and print data from input 1
        data = osc.get_data()
        ch1 = data['ch1']
        
        avg = sum(ch1)/len(ch1)

        # Move in the direction of greatest change
        if avg > last_avg: 
            print("Right Direction")
        else: 
            if (dir.name == 3):
                dir.value = 0
            else:
                # Python Version of dir.value++
                    values = list(Direction)
                    dir = values[(dir.value + 1) % len(values)] 

        last_avg = avg
        # print(ch1)


        if avg > 2:  # If the laser is in the absolute center of the target end the code
            print("Laser Locked")
            osc.relinquish_ownership()
            break


        match dir: # Make a movement based on the current direction
            case Direction.Up:
                send('CHL A1=1') # y axis
                send('REL A1 =100 G') # Move the laser UP
            case Direction.Right:
                send('CHL A1=0') # x axis
                send('REL A1 =100 G') # Move the laser RIGHT
            case Direction.Down:
                send('CHL A1=1') # y axis
                send('REL A1 =-100 G') # Move the laser DOWN
            case Direction.Left:
                send('CHL A1=0') # x axis
                send('REL A1 =-100 G') # Move the laser LEFT
        print(dir.name)


except KeyboardInterrupt: #Ctrl C to end
    print("Interrupted")
    osc.relinquish_ownership() # Make sure to relinquish or it will keep running on the moku
