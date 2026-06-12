"""Visualize where generate_samples() spawns its random starting cursors.

The training data in MakeSamples.py picks each random situation's starting
point from a rectangle centered on the beam peak. This script draws that
rectangle on top of the actual power scan so you can see the spawn area.

The geometry here mirrors generate_samples() exactly:
    - recenter the scan so the beam peak sits at (0, 0)
    - x_range / y_range = distance from center to the NEAREST scan edge
      on each axis (times edge_fraction)
    - starting points are uniform over [-x_range, x_range] x [-y_range, y_range]

Run:  python VisualizeSamples.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import LinearNDInterpolator

# ---- match the parameters used in LaserAlignment.py ----
NPZ_PATH = r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_att.npz'
EDGE_FRACTION = 1.0     # generate_samples default
MIN_POWER_FRAC = 0.01   # reject spawns below this fraction of peak power (0 = no filter)
N_SHOW = 2000           # how many accepted random spawn points to scatter
SEED = 7                # same seed LaserAlignment.py uses

np.random.seed(SEED)

# ---- load + recenter on the beam peak (same as generate_samples) ----
with np.load(NPZ_PATH) as data:
    x_scan = data['x_scan']
    y_scan = data['y_scan']
    power = data['power']

peak = np.argmax(power)
x_center = x_scan[peak]
y_center = y_scan[peak]
print(f'Beam center found at ({x_center:.0f}, {y_center:.0f})')

x_shifted = x_scan - x_center
y_shifted = y_scan - y_center

# ---- candidate bounds: the full scanned extent (no centered box) ----
x_lo, x_hi = float(x_shifted.min()) * EDGE_FRACTION, float(x_shifted.max()) * EDGE_FRACTION
y_lo, y_hi = float(y_shifted.min()) * EDGE_FRACTION, float(y_shifted.max()) * EDGE_FRACTION
print(f'Sampling extent: x=[{x_lo:.0f}, {x_hi:.0f}]  y=[{y_lo:.0f}, {y_hi:.0f}]')

# ---- power floor + interpolator, same as generate_samples ----
power_floor = float(power[peak]) * MIN_POWER_FRAC
print(f'Min readable spawn power: {power_floor:.3f} ({MIN_POWER_FRAC:.0%} of peak)')

interp = LinearNDInterpolator(np.column_stack([y_shifted, x_shifted]), power)

def get_power(x, y):
    result = interp([[y, x]])[0]
    return 0.0 if np.isnan(result) else float(result)

# ---- draw random spawn points, rejecting dark ones like generate_samples does ----
# Capped so an impossibly high floor (e.g. MIN_POWER_FRAC >= 1.0, where almost no
# point reaches peak power) plots what it found instead of looping forever.
xs, ys = [], []
attempt_budget = N_SHOW * 500
for _ in range(attempt_budget):
    if len(xs) >= N_SHOW:
        break
    x = np.random.uniform(x_lo, x_hi)
    y = np.random.uniform(y_lo, y_hi)
    if get_power(x, y) >= power_floor:
        xs.append(x)
        ys.append(y)
xs, ys = np.array(xs), np.array(ys)
if len(xs) < N_SHOW:
    print(f'WARNING: floor too high - found only {len(xs)}/{N_SHOW} spawns in '
          f'{attempt_budget} tries. Lower MIN_POWER_FRAC ({MIN_POWER_FRAC:.0%}).')

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 7))

# the real power scan as the background (colored by measured power)
sc = ax.scatter(x_shifted, y_shifted, c=power, s=8, cmap='viridis', alpha=0.9)
fig.colorbar(sc, ax=ax, label='power')

# outline the "readable" region: contour of the power field at the floor level.
# Spawns are only accepted inside this contour.
gx = np.linspace(x_lo, x_hi, 200)
gy = np.linspace(y_lo, y_hi, 200)
GX, GY = np.meshgrid(gx, gy)
GP = interp(np.column_stack([GY.ravel(), GX.ravel()])).reshape(GX.shape)
ax.contour(GX, GY, GP, levels=[power_floor], colors='orange', linewidths=2)
ax.plot([], [], color='orange', linewidth=2,
        label=f'readable boundary ({MIN_POWER_FRAC:.0%} peak)')  # legend proxy

# the candidate envelope = full scan extent; spawns are filtered by power, not this box
rect = Rectangle(
    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
    fill=False, edgecolor='red', linewidth=2, linestyle='--',
    label='scan extent (candidates)',
)
ax.add_patch(rect)

# the random spawn points + the beam center
ax.scatter(xs, ys, s=4, color='white', edgecolor='black',
           linewidths=0.2, alpha=0.5,
           label=f'random spawns (>= {MIN_POWER_FRAC:.0%} peak)')
ax.scatter([0], [0], marker='*', s=300, color='red',
           edgecolor='black', label='beam peak (0,0)', zorder=5)

ax.set_xlabel('x (picomotor steps, recentered on peak)')
ax.set_ylabel('y (picomotor steps, recentered on peak)')
ax.set_title('Random sample spawn area vs. measured beam')
ax.legend(loc='upper right')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('spawn_area.png', dpi=130)
print('Saved spawn_area.png')
plt.show()
