"""Deploy the trained model to align the laser on the real rig.

Rebuilds the model with the same simulated data as training (so the LinnModel
scaling matches), loads laser.weights.h5, then reads beam power off the Moku
oscilloscope and drives the picomotors until the beam is centered. Saves live
plots LaserMovement.png and IrisPower.png. Press Ctrl+C to stop safely.
"""

from moku.instruments import Oscilloscope
from moku.nn import LinnModel

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import os
import time
import numpy as np
import random

from XYMoveSamples import generate_samples
from picoMotor import send, setup, move, cleanup

# ============================== USER SETTINGS ==============================
n_samples = 20000   # must match LaserAlignment.py (used only to rebuild the scaling)
moves = 4           # length of the move history; must match LaserAlignment.py

SEED = 7            # MUST match LaserAlignment.py
OSC_IP = '192.168.1.167'   # Moku IP. Ethernet: 192.168.1.167   Wifi: 192.168.73.1

DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # this script's folder; scans live alongside it
# Must match TRAIN_SCANS in LaserAlignment.py (same scans, same order)
TRAIN_SCANS = [
    os.path.join(DATA_DIR, 'scan_data_iris1.1.npz'),
    os.path.join(DATA_DIR, 'scan_data_iris1.2.npz'),
    os.path.join(DATA_DIR, 'scan_data_iris2.0.npz'),
    os.path.join(DATA_DIR, 'scan_data_iris2.1.npz'),
    os.path.join(DATA_DIR, 'scan_data_att.npz'),
]
MODEL_DEFINITION = [(32, 'softsign'), (32, 'softsign'), (2, 'linear')]   # MUST match training
WEIGHTS_FILE = 'laser.weights.h5'

START_DRIVER = 1     # which mirror to align first (1 or 2)
VELOCITY = 100       # motor velocity during alignment
TIMEBASE = 0.005     # oscilloscope window: +-0.005 s around the trigger
PERCENTAGE = 0.03   # convergence tolerance (fraction of peak power)

# Per-mirror direction calibration [+X, +Y, -X, -Y], equalizing physical
# distance per direction. They are measured normal to Y+
RATIOS = {
    1: [1.74, 1.00, 1.53, 1.01],
    2: [1.39, 1.00, 1.46, 1.12],
}
# ===========================================================================

np.random.seed(SEED)
random.seed(SEED)

osc = Oscilloscope(OSC_IP, force_connect=True)

t_start = None   # set when the motor first moves (for the iris-power time box)

# ---------------------------------- Rebuild the model and load weights ----------------------------------
# Regenerate the training data so the LinnModel's input/output scaling matches,
# then load the trained weights instead of training again.
inputs, corrections = generate_samples(TRAIN_SCANS, n_samples, moves)

laser_mod = LinnModel()
out_dim = corrections.shape[1]
laser_mod.set_training_data(training_inputs=inputs, training_outputs=corrections)
laser_mod.construct_model(MODEL_DEFINITION, show_summary=False)
laser_mod.model.load_weights(WEIGHTS_FILE)


def read_power(driver):
    data = osc.get_data()
    return sum(data[f'ch{driver}']) / len(data[f'ch{driver}']) * 6.45   # multiplier to get power

def look_for_signal():
    # Expanding spiral search until the beam is found. legs: (axis, direction,
    # scale); scale equalizes physical distance per direction.
    if driver == 2:
        legs = [(0, 1, 1.39), (1, 1, 1), (0, -1, 1.46), (1, -1, 1.12)]
    elif driver == 1:
        legs = [(0, 1, 1.64), (1, 1, 1), (0, -1, 1.53), (1, -1, 1.01)]
    base_step = 50     # step size on the first ring (keep <= beam width)
    growth = 25        # step grows this much each leg, so the spiral expands faster
    seg_len = 1        # leg lengths grow 1,1,2,2,3,3,...
    leg = 0
    velocity = 1000
    send(f'vel a{driver} 0={velocity}')
    send(f'vel a{driver} 1={velocity}')

    while True:
        move_axis, move_dir, scale = legs[leg % 4]
        search_step = int((base_step + growth * leg) * scale)

        for _ in range(seg_len):
            search_power = read_power(driver)

            # Check before each move so a beam crossing isn't skipped mid-leg
            if search_power >= .05:   # found the beam
                velocity = 100
                send(f'vel a{driver} 0={velocity}')
                send(f'vel a{driver} 1={velocity}')
                xy_start()
                reset()
                return

            move(move_axis, move_dir, search_step, velocity, driver)

        leg += 1
        if leg % 2 == 0:   # grow the spiral every two legs
            seg_len += 1
        print('No signal - Scanning')

def fine_tune():
    # Optional hill-climb polish: nudge each axis +/- by a shrinking step, keeping
    # only improving moves. Call from the convergence block to enable.
    step = 30
    axis = 0
    while step >= 15:
        peak = read_power(driver)

        # Check +1 and keep it if it improves, else undo it and search -1
        move(axis, 1, step, velocity, driver)
        time.sleep(.1)
        cur_power = read_power(driver)
        if cur_power > peak:
            bearing = 1
            peak = cur_power
        else:
            move(axis, -1, step, velocity, driver)   # undo bad move
            bearing = -1

        while True:
            move(axis, bearing, step, velocity, driver)
            time.sleep(.1)
            cur_power = read_power(driver)
            if cur_power > peak:
                peak = cur_power   # new peak
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
    plt.colorbar(sc, ax=ax, label='Power (mW)')
    ax.scatter(X[0], Y[0], color='black', s=90, zorder=3, label='start')
    ax.scatter(X[-1], Y[-1], color='yellow', s=90, zorder=3, label='current')
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X-steps')
    ax.set_ylabel('Y-steps')
    ax.set_title('Laser beam path')
    plt.savefig('LaserMovement.png', dpi=150, bbox_inches='tight')

def plot_iris_power():
    # Per-iris start power (dot) vs. power after the model converges (orange bar)
    ORANGE, MAROON = '#E87722', '#861F41'   # Virginia Tech colors
    HALF, MUT, LW, ALPHA = 0.13, 15, 2.6, 0.6
    HEAD_FRAC = 0.4   # '-|>' head length = HEAD_FRAC * mutation_scale (points)

    prev = plt.gcf().number if plt.get_fignums() else None
    fig = plt.figure('iris_power')
    fig.clf()
    axes = fig.subplots(1, 2)

    # (name, start list, end list, leading cycles to skip)
    panels = (('Iris 1', iris1start, iris1end, 0),
              ('Iris 2', iris2start, iris2end, 0))

    # Pass 1: dots + dashes (centred on the cycle's x) + dotted max line
    for ax, (name, s, m, skip) in zip(axes, panels):
        ks = list(range(skip, min(len(s), len(m))))   # cycles to plot
        ys = []
        for i, k in enumerate(ks):
            x = i + 1
            ax.plot(x, s[k], 'o', color=MAROON, ms=9, zorder=5)            # start
            ax.plot([x - HALF, x + HALF], [m[k], m[k]], color=ORANGE,       # model
                    lw=4, solid_capstyle='round', zorder=3)
            ys += [s[k], m[k]]
        if ys:
            ax.axhline(max(ys), ls=':', color='gray', lw=1, zorder=1)       # max
        ax.set_title(name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Power (mW)')
        ax.grid(alpha=.3)
        if ks:
            # Cap the x-ticks so iteration labels don't bunch up; keep them integer
            ax.set_xlim(0.5, len(ks) + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    legend = [Line2D([], [], ls='', marker='o', color=MAROON, ms=9, label='Start'),
              Line2D([], [], color=ORANGE, lw=4, label='Neural Network')]
    fig.legend(handles=legend, loc='upper center', ncol=2, frameon=True,
               bbox_to_anchor=(0.5, 0.92))

    fig.suptitle('Iris Power During Alignment', y=0.97)

    # Elapsed-time box: starts when the motor first moved (t_start)
    if t_start is not None:
        mins, secs = divmod(int(time.time() - t_start), 60)
        fig.text(0.97, 0.95, f'Time: {mins:d}:{secs:02d}', ha='right', va='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white',
                                        edgecolor='gray'))

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.canvas.draw()   # finalize transforms before sizing arrows

    # Pass 2: the start -> model arrow, skipped when the change is shorter than
    # the arrow head. Convert head length from points to data units per panel.
    head_px = HEAD_FRAC * MUT * fig.dpi / 72.0
    for ax, (name, s, m, skip) in zip(axes, panels):
        inv = ax.transData.inverted()
        head_data = abs(inv.transform((0, head_px))[1] - inv.transform((0, 0))[1])
        ks = list(range(skip, min(len(s), len(m))))
        for i, k in enumerate(ks):
            x = i + 1
            if abs(m[k] - s[k]) >= head_data:   # start -> model (orange)
                ax.annotate('', xy=(x, m[k]), xytext=(x, s[k]),
                            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=LW,
                                            mutation_scale=MUT, shrinkA=0, shrinkB=0,
                                            alpha=ALPHA), zorder=4)

    fig.savefig('IrisPower.png', dpi=150, bbox_inches='tight')
    if prev is not None:
        plt.figure(prev)

def xy_start():
    # Fixed startup strafe: 4 seeding moves that give the model an initial history
    global axis, direction, step_size, pwr
    axis = [0, 1, 0, 1]
    direction = [1, 1, -1, -1]
    pwr = []
    steps = RATIOS[driver]   # [+X, +Y, -X, -Y]

    if read_power(driver) > .5:
        step_size = [50, 50, 50, 50]
    else:
        step_size = [100, 100, 100, 100]

    for i in range(moves):
        pwr.append(read_power(driver))
        move(axis[i], direction[i], int(steps[i] * 50), velocity, driver)

    # Reverse so slot [0] = most recent move
    axis      = axis[::-1]
    direction = direction[::-1]
    step_size = step_size[::-1]
    pwr       = pwr[::-1]

def reset():
    # Clear the per-iris plotting buffers
    global pos, Xmoves, Ymoves, Pmoves
    pos = [0, 0]
    Xmoves = []
    Ymoves = []
    Pmoves = []

try:
    velocity = VELOCITY
    setup(velocity)
    osc.set_timebase(0, TIMEBASE)

    driver = START_DRIVER
    t_start = time.time()   # motor first starts moving here
    xy_start()
    reset()

    peak = 0
    threshhold = .5
    percentage = PERCENTAGE

    iris1start = [read_power(driver)]
    iris1end = []
    iris2start = []
    iris2end = []

    plt.ion()   # interactive mode so the plot updates live without blocking moves

    while True:   # run the model and make the resulting moves

        ratio = RATIOS[driver]   # [+X, +Y, -X, -Y]
        if driver == 1 and len(iris1start) >= 2:
            peak = iris1start[-1]
            threshhold = iris1start[-1]
        elif driver == 2 and len(iris2start) >= 2:
            peak = iris2start[-1]
            threshhold = iris2start[-1]

        current_power = read_power(driver)
        time.sleep(.1)

        # Record the current location and the power there, so the path can be
        # coloured by power at each location
        Xmoves.append(pos[0])
        Ymoves.append(pos[1])
        Pmoves.append(current_power)

        move_set = np.array([[
            current_power,
            axis[0], direction[0], step_size[0], pwr[0],   # 1 move ago
            axis[1], direction[1], step_size[1], pwr[1],   # 2 moves ago
            axis[2], direction[2], step_size[2], pwr[2],   # 3 moves ago
            axis[3], direction[3], step_size[3], pwr[3],   # 4 moves ago
        ]])

        result = laser_mod.predict(move_set, scale=True, unscale_output=True)[0]

        step = round(result[1])   # step = the second part of the result

        # Alternate the axis from the last move
        move_axis = 1 if axis[0] == 0 else 0
        move_dir = 1 if result[0] > 0 else -1

        if move_axis == 0:
            pos[0] += step * move_dir
            print(f'The next move will be in the {"+X" if move_dir > 0 else "-X"} direction')
        else:
            pos[1] += step * move_dir
            print(f'The next move will be in the {"+Y" if move_dir > 0 else "-Y"} direction')

        plot_path()

        # Shift history back one slot; the most recent move goes into [0]
        axis =      [move_axis,     axis[0],      axis[1],      axis[2]]
        direction = [move_dir,      direction[0], direction[1], direction[2]]
        step_size = [step,          step_size[0], step_size[1], step_size[2]]
        pwr =       [current_power, pwr[0],       pwr[1],       pwr[2]]

        # Convergence condition: The power gradient is flat, power is above a threshold, and the last move increased power
        if (((max(pwr) - min(pwr)) <= percentage * peak) and (min(pwr) > (threshhold * (1 - percentage))) and (pwr[0] >= pwr[1])
                or (len(Xmoves) > 50)):

            if driver == 1:
                iris1end.append(read_power(driver))
            else:
                iris2end.append(read_power(driver))

            # fine_tune()   # uncomment to hill-climb polish each mirror after the model converges

            # Switch to the other mirror
            driver = 2 if driver == 1 else 1

            print(f'CENTER REACHED IN {len(Xmoves)} MOVES')

            if driver == 1:
                iris1start.append(read_power(driver))
            else:
                iris2start.append(read_power(driver))

            plot_iris_power()

            # STOP CONDITION: if the model's improvement (end - start) over the last three iris-2 iterations is below the threshold, alignment is done.
            if len(iris2end) >= 3 and len(iris2start) == len(iris2end):
                improvements = [e - s for s, e in zip(iris2start[-3:], iris2end[-3:])]
                if max(improvements) < 1.5 * percentage * (iris2end[-1]): 
                    plot_iris_power()   
                    print("LASER BEAM ALIGNMENT IS COMPLETE")
                    cleanup()
                    osc.relinquish_ownership()
                    break

            xy_start()
            reset()
            continue

        if current_power > peak:
            peak = current_power
        print(f'The peak power so far is {peak:.2f}')

        if max(pwr) < .05:
            look_for_signal()

        # Normalize the step based on the known per-direction offsets
        if   move_axis == 0 and move_dir == 1:  step = int(step * ratio[0])
        elif move_axis == 1 and move_dir == 1:  step = int(step * ratio[1])
        elif move_axis == 0 and move_dir == -1: step = int(step * ratio[2])
        elif move_axis == 1 and move_dir == -1: step = int(step * ratio[3])

        move(move_axis, move_dir, step, velocity, driver)

except KeyboardInterrupt:   # Ctrl+C to end
    cleanup()
    osc.relinquish_ownership()   # always relinquish or the next connection is hard
