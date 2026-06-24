"""Quick visualizer for the training samples.

Reproduces the per-sample walk from XYMoveSamples and draws each sample's move
history on top of the beam heatmap, printing the encoded model row too. Set
BALANCED = True to match the trainer's equal-samples-per-power-level quota.
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

# ============================== USER SETTINGS ==============================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # this script's folder
SCAN = os.path.join(DATA_DIR, 'scan_data_iris1.2.npz')
MOVES = 4
N_SHOW = 6              # trajectories to draw
N_SAMPLES = 2000       # quota target, used when BALANCED (matches a training run)
BIN_WIDTH = 0.01       # power-level width, must match generate_samples
PEAK_MIN, PEAK_MAX = 0.5, 3.0
MIN_POWER_FRAC = 0.005
BALANCED = True
SEED = 7
# ===========================================================================

random.seed(SEED)
np.random.seed(SEED)

# ---- load scan, center on the beam peak ----
with np.load(SCAN) as d:
    x_scan, y_scan, power = d['x_scan'], d['y_scan'], d['power']

peak = np.argmax(power)
xs = x_scan - x_scan[peak]
ys = y_scan - y_scan[peak]
peak_power = float(power[peak])

interp = LinearNDInterpolator(np.column_stack([ys, xs]), power)

def get_power(x, y):
    r = interp([[y, x]])[0]
    return 0.0 if np.isnan(r) else float(r)

x_lo, x_hi = int(np.floor(xs.min())), int(np.ceil(xs.max()))
y_lo, y_hi = int(np.floor(ys.min())), int(np.ceil(ys.max()))
power_floor = peak_power * MIN_POWER_FRAC

# ---- background heatmap of the beam ----
gx = np.linspace(x_lo, x_hi, 240)
gy = np.linspace(y_lo, y_hi, 240)
GX, GY = np.meshgrid(gx, gy)
HEAT = interp(np.column_stack([GY.ravel(), GX.ravel()])).reshape(GX.shape)


def balanced_spawns(n_samples, bin_width=BIN_WIDTH, pool_size=None):
    # Mirror of the equal-samples-per-power-level selection in XYMoveSamples;
    # returns spawn x, y, and the raw (unscaled) power at each spawn.
    if pool_size is None:
        pool_size = max(200_000, 80 * n_samples)

    px = np.random.randint(x_lo, x_hi + 1, pool_size)
    py = np.random.randint(y_lo, y_hi + 1, pool_size)
    pw = interp(np.column_stack([py, px]))
    keep = ~np.isnan(pw) & (pw >= power_floor)
    px, py, pw = px[keep], py[keep], pw[keep]

    uniq_pos, ui = np.unique(np.stack([px, py], axis=1), axis=0, return_index=True)
    uniq_pw = pw[ui]

    level_idx = np.floor(uniq_pw / bin_width).astype(int)
    levels = np.unique(level_idx)
    quota = max(1, round(n_samples / levels.size))

    sx, sy, sp = [], [], []
    for lv in levels:
        sel = np.where(level_idx == lv)[0]
        if sel.size >= quota:
            pick = np.random.choice(sel, size=quota, replace=False)
        else:
            extra = np.random.choice(sel, size=quota - sel.size, replace=True)
            pick = np.concatenate([sel, extra])
        sx.append(uniq_pos[pick, 0])
        sy.append(uniq_pos[pick, 1])
        sp.append(uniq_pw[pick])

    sx, sy, sp = np.concatenate(sx), np.concatenate(sy), np.concatenate(sp)
    order = np.random.permutation(sx.size)
    return sx[order], sy[order], sp[order]


def forward_walk(x0, y0):
    """Replicates the walk; returns positions and the encoded (reversed) lists."""
    x, y = x0, y0
    axis, direction, step_size, pwr = [], [], [], []
    pos = [(x, y)]                               # position before each move + final
    target_peak = 10 ** random.uniform(np.log10(PEAK_MIN), np.log10(PEAK_MAX))
    power_scale = target_peak / peak_power

    for j in range(MOVES):
        temp_axis = random.randint(0, 1) if j == 0 else (1 if axis[-1] == 0 else 0)
        pwr.append(get_power(x, y) * power_scale)         # power before this move

        if temp_axis == 0:
            temp_dir = -x / abs(x) if x != 0 else 1
            steps = abs(x + random.randint(-50, 50)) / 2
            x = x + steps if temp_dir == 1 else x - steps
        else:
            temp_dir = -y / abs(y) if y != 0 else 1
            steps = abs(y + random.randint(-50, 50)) / 2
            y = y + steps if temp_dir == 1 else y - steps

        axis.append(temp_axis)
        direction.append(temp_dir)
        step_size.append(steps)
        pos.append((x, y))

    current = get_power(x, y) * power_scale

    axis, direction, step_size, pwr = axis[::-1], direction[::-1], step_size[::-1], pwr[::-1]

    coord = y if axis[0] == 0 else x
    corr_steps = abs(coord) / 2
    corr_dir = 1 if coord < 0 else (-1 if coord > 0 else 0)

    row = [current]
    for k in range(MOVES):
        row.extend([axis[k], direction[k], step_size[k], pwr[k]])

    return pos, current, target_peak, row, (axis, direction, step_size, pwr), (coord, corr_steps, corr_dir)


# ---- choose spawns ----
if BALANCED:
    spawn_x, spawn_y, spawn_pw = balanced_spawns(N_SAMPLES)
    show = np.random.choice(spawn_x.size, N_SHOW, replace=False)
    spawns = [(int(spawn_x[k]), int(spawn_y[k])) for k in show]
    sel_label = f'balanced quota (target {N_SAMPLES}, {spawn_x.size} spawns)'
else:
    pool_x = np.random.randint(x_lo, x_hi + 1, 60000)
    pool_y = np.random.randint(y_lo, y_hi + 1, 60000)
    pool_p = interp(np.column_stack([pool_y, pool_x]))
    ok = ~np.isnan(pool_p) & (pool_p >= power_floor)
    pool_x, pool_y, pool_p = pool_x[ok], pool_y[ok], pool_p[ok]
    order = np.argsort(pool_p)
    idx = np.linspace(0, len(order) - 1, N_SHOW).astype(int)
    spawn_x = pool_x[order]; spawn_y = pool_y[order]; spawn_pw = pool_p[order]
    spawns = [(int(pool_x[order[k]]), int(pool_y[order[k]])) for k in idx]
    sel_label = 'sorted dim->bright spread'

# ---- trajectory grid ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

print('=' * 78)
print(f'Spawn selection: {sel_label}')
print('=' * 78)
axname = {0: 'X', 1: 'Y'}
for n, (sx, sy) in enumerate(spawns):
    pos, current, target_peak, row, hist, corr = forward_walk(sx, sy)
    coord, corr_steps, corr_dir = corr
    axis, direction, step_size, pwr = hist
    px = [p[0] for p in pos]; py = [p[1] for p in pos]

    ax = axes[n]
    ax.imshow(HEAT, extent=[x_lo, x_hi, y_lo, y_hi], origin='lower',
              cmap='viridis', aspect='auto', alpha=0.85)
    ax.plot(0, 0, marker='+', color='white', ms=14, mew=2)
    ax.plot(px, py, '-', color='white', lw=1.2, alpha=0.7, zorder=2)
    for k in range(MOVES):
        ax.annotate('', xy=(px[k + 1], py[k + 1]), xytext=(px[k], py[k]),
                    arrowprops=dict(arrowstyle='-|>', color='white', lw=1.6))
    ax.scatter(px[0], py[0], color='black', s=90, zorder=4, label='start (spawn)')
    ax.scatter(px[-1], py[-1], color='red', marker='*', s=200, zorder=5, label='current')

    if axis[0] == 0:
        ax.annotate('', xy=(px[-1], py[-1] + corr_dir * corr_steps), xytext=(px[-1], py[-1]),
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2.2, ls='--'))
    else:
        ax.annotate('', xy=(px[-1] + corr_dir * corr_steps, py[-1]), xytext=(px[-1], py[-1]),
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2.2, ls='--'))

    ax.set_title(f'#{n}: peak={target_peak:.2f} mW  current={current:.3f}\n'
                 f'start=({sx},{sy}) -> current=({px[-1]:.0f},{py[-1]:.0f})', fontsize=10)
    ax.set_xlabel('X steps'); ax.set_ylabel('Y steps')
    if n == 0:
        ax.legend(loc='upper right', fontsize=8)

    print(f'Sample #{n}  target_peak={target_peak:.3f} mW   current power {current:.4f}')
    for k in range(MOVES):
        print(f'  move[{k}] ({"most recent" if k == 0 else f"{k+1} ago":<11}): '
              f'axis={axname[axis[k]]} dir={direction[k]:+.0f} '
              f'step={step_size[k]:7.2f} pwr={pwr[k]:.4f}')
    print(f'  correction    : axis={axname[axis[0] ^ 1]} dir={corr_dir:+d} step={corr_steps:.2f}')
    print('-' * 78)

fig.suptitle(f'Forward-walk samples [{sel_label}]  black=spawn  red star=current  '
             'white=move history  red dashed=correction', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT = os.path.join(DATA_DIR, 'sample_visualization.png')
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print(f'Saved trajectories -> {OUT}')

# ---- distribution: balanced spawn (start) brightness vs where the walk ends ----
end_pw = np.empty(spawn_x.size)
for k in range(spawn_x.size):
    pos, *_ = forward_walk(int(spawn_x[k]), int(spawn_y[k]))
    end_pw[k] = get_power(pos[-1][0], pos[-1][1])     # raw (unscaled) power at current

fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
bins = np.linspace(power_floor, peak_power, 45)
axA.hist(spawn_pw, bins=bins, color='#4C72B0', edgecolor='white')
axA.set_title(f'Spawn (START) raw power\n{sel_label}')
axA.set_xlabel('power (mW)'); axA.set_ylabel('count')
axB.hist(end_pw, bins=bins, color='#C44E52', edgecolor='white')
axB.set_title('Current (END of walk) raw power')
axB.set_xlabel('power (mW)')
fig2.suptitle('Power distribution: balanced flat at the start, pulled toward the peak by the '
              'converging walk\n(model additionally scales each sample by a random peak in '
              f'[{PEAK_MIN}, {PEAK_MAX}] mW)', fontsize=11)
fig2.tight_layout(rect=(0, 0, 1, 0.93))
OUT2 = os.path.join(DATA_DIR, 'sample_distribution.png')
fig2.savefig(OUT2, dpi=130, bbox_inches='tight')
print(f'Saved distribution -> {OUT2}')
print(f'\nSpawn power : min {spawn_pw.min():.3f}  max {spawn_pw.max():.3f}  mean {spawn_pw.mean():.3f}')
print(f'Current pow : min {end_pw.min():.3f}  max {end_pw.max():.3f}  mean {end_pw.mean():.3f}')
