from moku.instruments import Oscilloscope

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time

from moku.nn import LinnModel
import numpy as np
import random

from XYMoveSamples import generate_samples
from picoMotor import send, setup, move, get_pos, cleanup

n_samples = 5000
moves = 4

SEED = 7 # Keep the same as Laser Alignment
np.random.seed(SEED)
random.seed(SEED)

osc = Oscilloscope('192.168.73.1', force_connect=True) #IP Address

# ----------------------------------------------------This rebuilds the model with the saved weights---------------------------

# Regenerate the same simulated situations LaserAlignment.py trained on, so the
# LinnModel's input/output scaling matches before we load the trained weights.
# Must match TRAIN_SCANS in LaserAlignment.py (same scans, same order).
TRAIN_SCANS = [
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris1.1.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris1.2.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris2.0.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris2.1.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_att.npz',
]
mod_counts, corrections = generate_samples(TRAIN_SCANS, n_samples, moves)

laser_mod = LinnModel()
out_dim = corrections.shape[1]
laser_mod.set_training_data(training_inputs=mod_counts, training_outputs=corrections)
model_definition = [(32, 'softsign'), (32, 'softsign'), (out_dim, 'linear')]   # MUST match training
laser_mod.construct_model(model_definition, show_summary=False)

# Load the trained weights instead of training
laser_mod.model.load_weights('laser.weights.h5')

def read_power():
    data = osc.get_data()
    return sum(data['ch1']) / len(data['ch1']) * 6.455

def look_for_signal():
    
    # (axis, direction, scale) - scale equalizes physical distance per direction
    # from measured 10-box strafe times: x+ 376, x- 396, y+ 271, y- 304
    if driver == 2:
        legs = [(0, 1, 1.39),   # +X 1.39   (Put offsets in the third spot)
                (1, 1, 1),      # +Y 1.00
                (0, -1, 1.46),  # -X 1.46
                (1, -1, 1.12)]  # -Y 1.12
    elif driver == 1:
         legs = [(0, 1, 1.64),  # +X 1.84   (Put offsets in the third spot)
                (1, 1, 1),      # +Y 1.00
                (0, -1, 1.53),  # -X 1.53
                (1, -1, 1.01)]  # -Y 1.01
    base_step = 50     # step size on the first ring (keep <= beam width so it isn't skipped)
    growth = 25        # step grows this much each leg, so the spiral expands faster
    seg_len = 1         # leg lengths grow 1,1,2,2,3,3,...
    leg = 0
    velocity = 1000
    send(f'vel a{driver} 0={velocity}')
    send(f'vel a{driver} 1={velocity}')

    while True:

        move_axis, move_dir, scale = legs[leg % 4]
        search_step = int((base_step + growth * leg) * scale)   # scaled so each direction covers equal distance

        for _ in range(seg_len):

            search_power = read_power()
            # Plot the spiral scan
            """
            Xmoves.append(pos[0])
            Ymoves.append(pos[1])
            Pmoves.append(search_power)
            plot_path()
            """

            # Check before each move so a beam crossing isn't skipped mid-leg.
            if search_power >= .01:   # Found the beam
                velocity = 100
                send(f'vel a{driver} 0={velocity}')
                send(f'vel a{driver} 1={velocity}')
                xy_start()
                reset()
                return

            move(move_axis, move_dir, search_step, velocity, driver)

        leg += 1
        if leg % 2 == 0:               # Grow the spiral every two legs
            seg_len += 1
        print('No signal - Scanning')

def fine_tune():
    step = 30
    axis = 0
    while step >= 15:
        peak = read_power()

        # Check + 1 and keep it if it improves, else undo it and search - 1
        move(axis, 1, step, velocity, driver)
        time.sleep(.1)
        cur_power = read_power()
        if cur_power > peak:
            bearing = 1
            peak = cur_power
        else:
            move(axis, -1, step, velocity, driver) # Undo bad move
            bearing = -1

        while True:
            move(axis, bearing, step, velocity, driver)
            time.sleep(.1)
            cur_power = read_power()
            if cur_power > peak:
                peak = cur_power # New peak
            else:
                move(axis, -bearing, step, velocity, driver)
                break

        # Switch axis each round, halve the step after both axes are done
        if axis == 0:
            axis = 1
        else:
            axis = 0
            step //= 2

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

def plot_iris_power():
    # Per alignment cycle, show the beam power at three stages as a pair of
    # arrows:  start --(orange)--> model --(maroon)--> tuned
    #   start = before the model scan begins
    #   model = after the model converges, before fine-tuning  (iris*end)
    #   tuned = after fine_tune() peaks it
    # Iris 1 and iris 2 get their own panel since they sit at different powers.
    # Both arrows share the cycle's x and are drawn translucent so you can see
    # both where they overlap. An arrow whose change is smaller than its own head
    # is dropped (just the dash is shown) so tiny moves aren't all-head stubs.
    # A dotted gray line marks each iris's max power reached.
    GRAY, ORANGE, MAROON = '#75787B', '#E87722', '#861F41'   # Virginia Tech
    HALF, MUT, LW, ALPHA = 0.13, 15, 2.6, 0.6
    HEAD_FRAC = 0.4   # '-|>' head length = HEAD_FRAC * mutation_scale (points)

    prev = plt.gcf().number if plt.get_fignums() else None   # restore later so
    fig = plt.figure('iris_power')                           # plot_path()'s
    fig.clf()                                                # current-figure
    axes = fig.subplots(1, 2)                                # isn't redirected

    # last field = how many leading cycles to skip
    panels = (('Iris 1', iris1start, iris1end, iris1tuned, 0),
              ('Iris 2', iris2start, iris2end, iris2tuned, 0))

    # Pass 1: dots + dashes (centred on the cycle's x) + dotted max line.
    for ax, (name, s, m, t, skip) in zip(axes, panels):
        ks = list(range(skip, min(len(s), len(m), len(t))))   # cycles to plot
        ys = []
        for i, k in enumerate(ks):
            x = i + 1
            ax.plot(x, s[k], 'o', color=GRAY, ms=9, zorder=5)            # start
            ax.plot([x - HALF, x + HALF], [m[k], m[k]], color=ORANGE,    # model
                    lw=4, solid_capstyle='round', zorder=3)
            ax.plot([x - HALF, x + HALF], [t[k], t[k]], color=MAROON,    # tuned
                    lw=4, solid_capstyle='round', zorder=3)
            ys += [s[k], m[k], t[k]]
        if ys:
            ax.axhline(max(ys), ls=':', color='gray', lw=1, zorder=1)    # max
        ax.set_title(name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Power (mW)')
        ax.grid(alpha=.3)
        if ks:
            ax.set_xticks(range(1, len(ks) + 1))

    legend = [Line2D([], [], ls='', marker='o', color=GRAY, ms=9, label='Start'),
              Line2D([], [], color=ORANGE, lw=4, label='Neural Network'),
              Line2D([], [], color=MAROON, lw=4, label='Fine Tune')]
    fig.legend(handles=legend, loc='upper center', ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 0.92))   # key above the panels, out of the way

    fig.suptitle('Iris Power During Alignment', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.canvas.draw()   # finalize transforms (axis sizes) before sizing arrows

    # Pass 2: translucent overlapping arrows, skipped when the change is shorter
    # than the arrow head. Convert the head length from points to data units per
    # panel, since the two panels span very different power ranges.
    head_px = HEAD_FRAC * MUT * fig.dpi / 72.0
    for ax, (name, s, m, t, skip) in zip(axes, panels):
        inv = ax.transData.inverted()
        head_data = abs(inv.transform((0, head_px))[1] - inv.transform((0, 0))[1])
        ks = list(range(skip, min(len(s), len(m), len(t))))
        for i, k in enumerate(ks):
            x = i + 1
            if abs(m[k] - s[k]) >= head_data:   # start -> model (orange)
                ax.annotate('', xy=(x, m[k]), xytext=(x, s[k]),
                            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=LW,
                                            mutation_scale=MUT, shrinkA=0, shrinkB=0,
                                            alpha=ALPHA), zorder=4)
            if abs(t[k] - m[k]) >= head_data:   # model -> tuned (maroon)
                ax.annotate('', xy=(x, t[k]), xytext=(x, m[k]),
                            arrowprops=dict(arrowstyle='-|>', color=MAROON, lw=LW,
                                            mutation_scale=MUT, shrinkA=0, shrinkB=0,
                                            alpha=ALPHA), zorder=4)

    fig.savefig('IrisPower.png', dpi=150, bbox_inches='tight')
    if prev is not None:
        plt.figure(prev)

def xy_start():
    # 4 moves and data to start with
    global axis, direction, pwr
    axis = [0,1,0,1]
    direction = [1,1,-1,-1]
    pwr = []
    if driver == 1:
        steps = [1.84, 1.00, 1.53, 1.01] # [+X, +Y, -X, -Y]
    elif driver == 2:
        steps = [1.39, 1.00, 1.46, 1.12] # [+X, +Y, -X, -Y]

    for i in range(moves):
        pwr.append(read_power())
        move(axis[i], direction[i], (int)(steps[i]*25), velocity, driver)

    # Reverse so slot [0] = most recent move 
    axis      = axis[::-1]
    direction = direction[::-1]
    pwr       = pwr[::-1]

def reset():
    global pos, Xmoves, Ymoves, Pmoves
    # For graphing
    pos = [0,0]
    Xmoves = []
    Ymoves = []
    Pmoves = []

try:
    velocity = 100
    setup(velocity)

    osc.set_timebase(0, 0.001) # Define the length of data you get with respect to time (+- .001 seconds from trigger point)

    driver = 1
    xy_start()
    reset()

    peak = 0

    iris1start = [read_power()]
    iris1end = []
    iris1tuned = []
    iris2start = []
    iris2end = []
    iris2tuned = []

    plt.ion()   # interactive mode so the plot updates live without blocking moves

    while True: # Run the model and make the resulting moves

        if driver == 1:
            ratio = [1.84, 1.00, 1.53, 1.01] # [+X, +Y, -X, -Y]
        elif driver == 2:
            ratio = [1.39, 1.00, 1.46, 1.12] # [+X, +Y, -X, -Y]
        
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
        axis[2], direction[2], pwr[2],   # 3 moves ago
        axis[3], direction[3], pwr[3],   # 4 moves ago
        ]])

        result = laser_mod.predict(move_set, scale=True, unscale_output=True)[0] 

        # Step = the second part of the result
        step = round(result[1])

        # Alternate the axis from the last move
        if axis[0] == 0:
            move_axis = 1
        else:
            move_axis = 0
        move_dir = 1 if result[0] > 0 else -1

        if move_axis == 0:
            pos[0] += step * move_dir
            print(f'The next move will be in the {"+X" if move_dir > 0 else "-X"} direction')
        else:
            pos[1] += step * move_dir
            print(f'The next move will be in the {"+Y" if move_dir > 0 else "-Y"} direction')

        plot_path()

        # Shift history back one slot, the most recent move goes into [0]
        axis =      [move_axis,     axis[0],      axis[1],      axis[2]]
        direction = [move_dir,      direction[0], direction[1], direction[2]]
        pwr =       [current_power, pwr[0],       pwr[1],       pwr[2]]

        # If it doesn't find a new max, get back to the max
        
        if (((max(pwr) - min(pwr)) <= .04*peak) and (min(pwr) > .1) or (len(Xmoves) > 50)): # Change power meter condition

            if driver == 1: iris1end.append(read_power())
            else: iris2end.append(read_power())

            fine_tune()

            if driver == 1:
                iris1tuned.append(read_power())
                driver = 2
            else:
                iris2tuned.append(read_power())
                driver = 1

            print(f'CENTER REACHED IN {len(Xmoves)} MOVES, PLEASE MOVE THE METER TO IRIS {driver}')
            time.sleep(10)

            if driver == 1: iris1start.append(read_power())
            else: iris2start.append(read_power())

            plot_iris_power()

            xy_start()
            reset()
            continue

        if current_power > peak:
            peak = current_power
        print(f'The peak power so far is {peak:.2f}')

        if max(pwr) < .01:
            look_for_signal()   

        # This block normalizes the steps based on their known offsets
        if   move_axis == 0 and move_dir == 1:  step = (int)(step*ratio[0])
        elif move_axis == 1 and move_dir == 1:  step = (int)(step*ratio[1])
        elif move_axis == 0 and move_dir == -1: step = (int)(step*ratio[2])
        elif move_axis == 1 and move_dir == -1: step = (int)(step*ratio[3])

        move(move_axis, move_dir, step, velocity, driver) # 1 for first mirror, 2 for second mirror

except KeyboardInterrupt: # Ctrl C to end
    cleanup()
    osc.relinquish_ownership() # Make sure to relinquish or it will be hard to connect next time