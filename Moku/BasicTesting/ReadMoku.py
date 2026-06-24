"""Live readout of the Moku oscilloscope channel-1 average.

Continuously reads input 1, plots its running average over time, and prints the
power (V x 6.455). Handy for checking the beam signal. Press Ctrl+C to stop.
"""

from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt
import time

# ============================== USER SETTINGS ==============================
OSC_IP = '192.168.73.1'   # Moku IP address
TIMEBASE = 0.001          # oscilloscope window (+- seconds around the trigger)
# ===========================================================================

osc = Oscilloscope(OSC_IP, force_connect=True)

avg_data = []
time_data = []

plt.ion()   # makes it easier to work with the graphs

try:
    start_time = time.time()
    osc.set_timebase(0, TIMEBASE)

    last_avg = 0

    while True:   # continuously read and print data from input 1

        data = osc.get_data()
        ch1 = data['ch1']

        avg = sum(ch1)/len(ch1)
        change = avg - last_avg   # could serve as an incentive signal (more change = closer)
        last_avg = avg

        time_data.append(time.time() - start_time)   # seconds since start
        avg_data.append(avg)

        plt.clf()
        plt.plot(time_data, avg_data, '-')   # '-' for line, '.' for dots
        plt.xlabel('Approximate Time (s)')
        plt.ylabel('Input Voltage (V)')
        plt.title('Channel 1 Average Signal')
        plt.pause(0.01)

        print(avg*6.455)

except KeyboardInterrupt:   # Ctrl+C to end
    print("Done")
    osc.relinquish_ownership()   # always relinquish or the next connection is hard
