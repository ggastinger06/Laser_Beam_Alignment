"""Shared data-generation helpers for the laser-alignment scripts.

Both LaserAlignment.py (training) and PhysicalAlignment.py (deployment) import
generate_samples from here so the simulated training situations - and therefore
the LinnModel input/output scaling derived from them - stay identical between
the two. Keeping this in one place avoids the two copies drifting apart.
"""

import numpy as np
import random
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

# min_power_frac defines the boundary that the random points can spawn in
def generate_samples(npz_path, n_samples, moves, step, min_power_frac=0.01, verbose=True):
  
    with np.load(npz_path) as data:
        x_scan = data['x_scan']
        y_scan = data['y_scan']
        power = data['power']

    # Find the highest power and shift so beam center is at (0, 0)
    peak = np.argmax(power)
    x_center = x_scan[peak]
    y_center = y_scan[peak]
    if verbose:
        print(f'Beam center found at ({x_center:.0f}, {y_center:.0f})')

    x_shifted = x_scan - x_center
    y_shifted = y_scan - y_center

    # Interpolator to fill in data gaps
    interp = LinearNDInterpolator(
        np.column_stack([y_shifted, x_shifted]), power
    )

    def get_power(x, y):
        result = interp([[y, x]])[0]  # adding [0] makes it a scalar
        if np.isnan(result):
            return 0.0
        return float(result)

    # Set the bounds to the whole scan area
    x_lo, x_hi = float(x_shifted.min()), float(x_shifted.max())
    y_lo, y_hi = float(y_shifted.min()), float(y_shifted.max())
    if verbose:
        print(f'Sampling extent: x=[{x_lo:.0f}, {x_hi:.0f}]  y=[{y_lo:.0f}, {y_hi:.0f}]')

    # Spawns dimmer than this floor are rejected so every cursor starts on real signal
    power_floor = float(power[peak]) * min_power_frac
    if verbose:
        print(f'Min readable spawn power: {power_floor:.3f} ({min_power_frac:.0%} of peak)')

    # -------------------------------- Create Random Sample Situations --------------------------------

    mod_counts = np.zeros((n_samples, 1 + moves * 3))  # Last data measurements and current power
    corrections = np.zeros((n_samples, 2))             # next move toward center [x, y]

    for i in tqdm(range(n_samples)):  # NOTE: the center of the beam is at (0, 0)

        # Draw a point from the scanned area who's power is greater than the threshold 
        current = -np.inf
        while current < power_floor:
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            current = get_power(x, y)

        # old symmetric-rectangle spawn
        # x = np.random.uniform(-x_range, x_range)
        # y = np.random.uniform(-y_range, y_range)

        axis = []
        direction = []
        pwr = []

        xp = x
        yp = y

        for j in range(moves):  # create random last moves
            temp_axis = random.randint(0, 1)
            temp_dir = random.choice([-1, 1])

            # Step backwards to find where we were before
            if temp_axis == 0:
                xp -= temp_dir * step
            else:
                yp -= temp_dir * step

            axis.append(temp_axis)
            direction.append(temp_dir)
            pwr.append(get_power(xp, yp))  # power at that earlier position

        row = [current]                          # current power state
        for j in range(moves):                   # (j+1) moves ago
            row.extend([axis[j], direction[j], pwr[j]])
        mod_counts[i] = row

        # clockwise
        test_powers= [
            get_power(x, y + step),
            get_power(x + step, y),
            get_power(x, y - step),
            get_power(x - step, y),
        ]
        if max(test_powers) == test_powers[0]:
            corrections[i] = [0, 1]
        elif max(test_powers) == test_powers[1]:
            corrections[i] = [1, 0]
        elif max(test_powers) == test_powers[2]:
            corrections[i] = [0, -1]
        elif max(test_powers) == test_powers[3]:
            corrections[i] = [-1, 0]

        
        """
        if abs(x) > abs(y):
            corrections[i] = [-np.sign(x), 0]
        else:
            corrections[i] = [0, -np.sign(y)]
        """

    return mod_counts, corrections
