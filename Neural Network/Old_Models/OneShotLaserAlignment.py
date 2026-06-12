# import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from copy import copy
from tqdm import tqdm
import tensorflow as tf128

from moku.nn import LinnModel, save_linn

from MultiAdjust.emitter_simulator import QuantumEmitter

# Amount of random cursors to simulate with
n_samples = 2500
offset = 50



# Generate/import training data here 
# --- Beam parameters ---
beam_sigma   = 400    # width of Gaussian in steps — tune to match your real beam
peak_power   = 1.0
noise_level  = 0.01

# --- Scan grid — matches your training range of ±1000 ---
x_positions = np.arange(-1200, 1250, 20)    # x resolution within each row
y_positions = np.arange(-600,   650, 100)   # row spacing

x_scan = []
y_scan = []
power  = []

for y in y_positions:
    for x in x_positions:
        x_scan.append(float(x))
        y_scan.append(float(y))

        # 2D Gaussian centered at (0, 0)
        p = peak_power * np.exp(
            -(x**2 + y**2) / (2 * beam_sigma**2)
        )
        p += np.random.normal(0, noise_level)   # small measurement noise
        power.append(float(np.clip(p, 0, None)))

x_scan = np.array(x_scan)
y_scan = np.array(y_scan)
power  = np.array(power)



# Correct — uses the original position arrays
power_grid = power.reshape(len(y_positions), len(x_positions))
interp = RegularGridInterpolator(
    (y_positions, x_positions),
    power_grid,
    method='linear',
    bounds_error=False,
    fill_value=0.0
)


counts = np.zeros(n_samples)
mod_counts = np.zeros((n_samples, 4)) # x and y, positive and negative derivatives
corrections = np.zeros((n_samples, 2)) # needed change to get to (0,0)


points = np.column_stack([x_scan, y_scan])
values = np.array(power)

def get_power(x, y):
    return float(interp([[y, x]])[0])
    """
    result = griddata(points, values, (x, y), method='linear')
    if np.isnan(result): # If no data
        return 0.0
    return float(result)
    """


def sim_move(axis, steps, direction, current_x, current_y):
    instances = int(steps / 100)
    values = []

    x, y = current_x, current_y

    for i in range(instances):
        # Step position in the right axis and direction
        if axis == 'x':
            x += 100 * direction
        else:
            y += 100 * direction

        point = get_power(x, y)
        values.append(point)

    return values



for i in tqdm(range(n_samples)): # Makes this many samples. NOTE: The center of the beam is centered at 0,0

    # Generate a random starting point
    x = np.random.uniform(-1000, 1000) 
    y = np.random.uniform(-500, 500)

    pxp = sim_move('x', 500, 1, x, y)
    pxn = sim_move('x', 500, -1, x, y)
    pyp = sim_move('y', 500, 1, x, y)
    pyn = sim_move('y', 500, -1, x, y)
    
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
    
    mod_counts[i] = [xn, xp, yn, yp] #this will be the change in x and the change in y
    

    corrections[i] = [-x / 1000, -y / 1000] # Finer data adjustments

# Training
laser_mod = LinnModel()
laser_mod.set_training_data(training_inputs=mod_counts, training_outputs=corrections)
model_definition = [(100, 'relu'),(100, 'relu'), (64, 'relu'), (64, 'relu'), (2,'linear')]
laser_mod.construct_model(model_definition, show_summary=True)
history = laser_mod.fit_model(epochs=500, es_config={'patience':50, 'restore':True}, validation_split=0.1)


# plot the loss and validation loss as a function of epoch to see how our training went.
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()
 
preds = laser_mod.predict(mod_counts, scale=True, unscale_output=True)