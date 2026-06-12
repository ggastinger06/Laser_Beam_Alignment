from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt
from matplotlib import cm
import time

from moku.nn._linn import WeightClip as _WeightClip
from moku.nn import LinnModel
import numpy as np
import random

from ThreeMoveSamples import generate_samples
from MultiAdjust.picoMotor import send, setup, move, get_pos, cleanup


n_samples = 5000
step = 50
moves = 2 
velocity = 100

SEED = 7 # Keep the same as Laser Alignment
np.random.seed(SEED)
random.seed(SEED)

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# ----------------------------------------------------This rebuilds the model with the saved weights---------------------------

# Regenerate the same simulated situations LaserAlignment.py trained on, so the
# LinnModel's input/output scaling matches before we load the trained weights.
mod_counts, corrections = generate_samples(
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data3.npz',
    n_samples, moves, step,
)

laser_mod = LinnModel()
out_dim = corrections.shape[1]
laser_mod.set_training_data(training_inputs=mod_counts, training_outputs=corrections)
model_definition = [(32, 'softsign'), (32, 'softsign'), (out_dim, 'linear')]   # MUST match training
laser_mod.construct_model(model_definition, show_summary=False)

# Load the trained weights instead of training
laser_mod.model.load_weights('laser_levels.weights.h5')

def read_power():
    data = osc.get_data()
    return sum(data['ch1']) / len(data['ch1']) * 6

def look_for_signal():
    # Square-spiral search outward from the current position until the beam is
    # reacquired. Updates current_power and the move history (axis/direction/pwr)
    # so the model has fresh data when control returns to the main loop.
    global current_power, axis, direction, pwr

    # (axis, direction, scale) - scale equalizes physical distance per direction
    # from measured 10-box strafe times: x+ 376, x- 396, y+ 271, y- 304
    legs = [(0, 1, 1.39),   # +X 1.39
            (1, 1, 1.00),   # +Y 1.00
            (0, -1, 1.46),  # -X 1.46
            (1, -1, 1.12)]  # -Y 1.12
    base_step = 100     # step size on the first ring (keep <= beam width so it isn't skipped)
    growth = 50        # step grows this much each leg, so the spiral expands faster
    seg_len = 1         # leg lengths grow 1,1,2,2,3,3,...
    leg = 0
    velocity = 1000
    send(f'vel a1 0={velocity}') 
    send(f'vel a1 1={velocity}') 

    while current_power < .01: # Need at least .01 mV

        move_axis, move_dir, scale = legs[leg % 4]
        search_step = int((base_step + growth * leg) * scale)   # scaled so each direction covers equal distance

        for _ in range(seg_len):
            # Record this move + the power before it, newest in slot [0]
            axis = [move_axis, axis[0], axis[1]]
            direction = [move_dir,  direction[0], direction[1]]
            pwr = [current_power, pwr[0], pwr[1]]

            move(move_axis, move_dir, search_step, velocity)
            current_power = read_power()

            if current_power >= .01:   # Found the beam
                velocity = 100
                send(f'vel a1 0={velocity}') 
                send(f'vel a1 1={velocity}') 
                return
            if seg_len > 40:           # Safety cap on spiral radius
                print('Spiral search finished - no beam found')
                return

        leg += 1
        if leg % 2 == 0:               # Grow the spiral every two legs
            seg_len += 1
        print('No signal - Scanning')

def plot_path():
    X = np.array(Xmoves, dtype=float)
    Y = np.array(Ymoves, dtype=float)
    P = np.array(Pmoves, dtype=float)

    plt.clf()
    ax = plt.gca()
    # Connecting line so the order/trail of the path stays visible
    ax.plot(X, Y, '-', color='gray', linewidth=2, zorder=1)
    # Colour each visited location by the power measured there
    sc = ax.scatter(X, Y, c=P, cmap='viridis', s=45, zorder=2)
    plt.colorbar(sc, ax=ax, label='power')
    ax.scatter(X[0], Y[0], color='black', s=90, zorder=3, label='start')
    ax.scatter(X[-1], Y[-1], color='yellow', s=90, zorder=3, label='current')
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X-steps')
    ax.set_ylabel('Y-steps')
    ax.set_title('Laser beam path')
    plt.savefig('LaserMovement.png', dpi=150, bbox_inches='tight')

def three_move_start():
    # 3 moves and data to start with
    global axis, direction, pwr
    axis = [1,0,1]
    direction = [1,1,-1]
    pwr = []

    for i in range(3):
        pwr.append(read_power())
        move(axis[i], direction[i], step, velocity)

    # Reverse so slot [0] = most recent move 
    axis      = axis[::-1]
    direction = direction[::-1]
    pwr       = pwr[::-1]

setup(velocity)

try:
    
    osc.set_timebase(0, 0.001) # Define the length of data you get with respect to time (+- .001 seconds from trigger point)

    three_move_start()

    # Reverse so slot [0] = most recent move 
    axis      = axis[::-1]
    direction = direction[::-1]
    pwr       = pwr[::-1]

    peak = 0
    peak_count = 0

    # For graphing
    pos = [0,0]
    Xmoves = []
    Ymoves = []
    Pmoves = []

    plt.ion()   # interactive mode so the plot updates live without blocking moves

    while True: # Run the model and make the resulting moves
        
        current_power = read_power()
        time.sleep(.1)

        # Record the beam's current location and the power measured there, so the
        # path can be coloured by power at each visited location
        Xmoves.append(pos[0])
        Ymoves.append(pos[1])
        Pmoves.append(current_power)

        move_set = np.array([[
        current_power,                   
        axis[0], direction[0], pwr[0],   # 1 move ago
        axis[1], direction[1], pwr[1],   # 2 moves ago
        axis[2], direction[2], pwr[2]    # 3 moves ago
        ]])

        result = laser_mod.predict(move_set, scale=True, unscale_output=True)[0]  # result = [0,1],[0,-1],[1,0], or [-1,0] 

        # Figure out what axis and direction the move was
        if abs(result[0]) > abs(result[1]): 
            move_axis = 0                                
            if result[0] > 0: 
                move_dir = 1 
                print('The next move will be in the +X direction')
                pos[0] += step
            else: 
                move_dir = -1
                print('The next move will be in the -X direction')
                pos[0] -= step
        else:
            move_axis = 1                   
            if result[1] > 0: 
                move_dir = 1 
                print('The next move will be in the +Y direction')
                pos[1] += step
            else: 
                move_dir = -1
                print('The next move will be in the -Y direction')
                pos[1] -= step

        plot_path()

        # Shift history back one slot, the most recent move goes into [0]
        axis = [move_axis, axis[0], axis[1]]
        direction = [move_dir, direction[0], direction[1]]
        pwr = [current_power, pwr[0], pwr[1]]

        print(f'Amount of moves without reaching the max power of {peak:.2f}: {peak_count}')  
        # If it doesn't find a new max, get back to the max
        if peak_count > 5 and current_power >= peak*.99:
            cleanup()
            osc.relinquish_ownership()
            break

        if current_power > peak:
            peak = current_power
            peak_count = 0
        else:
            peak_count += 1

        if max(pwr) < .01:
            look_for_signal()

        move(move_axis, move_dir, step, velocity)

except KeyboardInterrupt: #nCtrl C to end
    cleanup()
    osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time