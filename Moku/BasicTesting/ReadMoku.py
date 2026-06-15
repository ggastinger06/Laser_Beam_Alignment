from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt

import time

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# Store data in an array
avg_data = []
time_data = [] 

plt.ion() # Makes it earier to work with the graphs

try:
    start_time = time.time()  # record start time
    
    osc.set_timebase(0,.001) #(+- .001 seconds from trigger point)

    last_avg = 0

    while True: # Continuously read and print data from input 1

        data = osc.get_data()
        ch1 = data['ch1']

        avg = sum(ch1)/len(ch1)
        change = avg - last_avg # Change could be used to be an incentive for the ai (More change, closer to spot)
        last_avg = avg

        """
        if avg > 1: #Look for any indication of the laser hitting the target
            print("Detected")
        else:
            print("Not found")
        """
        
        time_data.append(time.time() - start_time)  # seconds since start
        avg_data.append(avg)
        
        
        # print(avg)    
        plt.clf()
        plt.plot(time_data, avg_data, '-')  # '-' for line, '.' for dots
        plt.xlabel('Approximate Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title('Channel 1 Average Signal')
        plt.pause(0.01)
       
        print(avg*6.455)

except KeyboardInterrupt: #Ctrl C to end
    print("Done")
    osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time