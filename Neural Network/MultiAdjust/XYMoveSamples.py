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
def generate_samples(npz_paths, n_samples, moves, step, min_power_frac=0.005,
                     bin_width=0.01, pool_size=None, verbose=True):
    """Generate training rows from one or more beam scans.

    npz_paths is a single path or a list of paths. n_samples is the total
    across all scans: each scan contributes an equal share so no single scan
    dominates training. The combined rows are shuffled so scans are
    interleaved (otherwise Keras' validation_split would carve the validation
    set entirely out of the last scan).
    """
    if isinstance(npz_paths, str):
        npz_paths = [npz_paths]

    per_scan = max(1, round(n_samples / len(npz_paths)))

    all_counts, all_corrections = [], []
    for path in npz_paths:
        if verbose:
            print(f'--- {per_scan} samples from {path} ---')
        counts, corr = _generate_from_scan(
            path, per_scan, moves, step, min_power_frac=min_power_frac,
            bin_width=bin_width, pool_size=pool_size, verbose=verbose,
        )
        all_counts.append(counts)
        all_corrections.append(corr)

    mod_counts = np.concatenate(all_counts)
    corrections = np.concatenate(all_corrections)

    order = np.random.permutation(mod_counts.shape[0])
    return mod_counts[order], corrections[order]


def _generate_from_scan(npz_path, n_samples, moves, step, min_power_frac=0.005,
                        bin_width=0.01, pool_size=None, verbose=True):

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

    # Set the bounds to the whole scan area (integer motor steps)
    x_lo, x_hi = int(np.floor(x_shifted.min())), int(np.ceil(x_shifted.max()))
    y_lo, y_hi = int(np.floor(y_shifted.min())), int(np.ceil(y_shifted.max()))
    if verbose:
        print(f'Sampling extent: x=[{x_lo}, {x_hi}]  y=[{y_lo}, {y_hi}]')

    # Spawns dimmer than this floor are rejected so every cursor starts on real signal
    power_floor = float(power[peak]) * min_power_frac
    if verbose:
        print(f'Min readable spawn power: {power_floor:.3f} ({min_power_frac:.0%} of peak)')

    # -------------------------------- Equal samples per power level --------------------------------
    # The universe of "possible points" is the set of integer motor-step
    # positions inside the scan area (the picomotors move in whole steps). We
    # draw one large candidate pool, keep the points above the floor, and dedupe
    # to unique integer positions. Each unique position is binned into a power
    # level of width bin_width, aligned to absolute multiples of bin_width so the
    # levels fall on round values (..., 2.08, 2.09, 2.10, ...). Every level then
    # contributes the same quota of spawns. Unique points are used first; a level
    # only duplicates spawns once all of its possible points are already in use.

    if pool_size is None:
        pool_size = max(200_000, 80 * n_samples)

    px = np.random.randint(x_lo, x_hi + 1, pool_size)
    py = np.random.randint(y_lo, y_hi + 1, pool_size)
    pw = interp(np.column_stack([py, px]))            # vectorized power lookup

    keep = ~np.isnan(pw) & (pw >= power_floor)
    px, py, pw = px[keep], py[keep], pw[keep]

    # Unique integer positions = the possible spawn points actually discovered
    uniq_pos, ui = np.unique(np.stack([px, py], axis=1), axis=0, return_index=True)
    uniq_pw = pw[ui]

    # Bin every unique position into an absolute-aligned power level
    level_idx = np.floor(uniq_pw / bin_width).astype(int)
    levels = np.unique(level_idx)
    n_levels = levels.size
    quota = max(1, round(n_samples / n_levels))
    if verbose:
        print(f'{n_levels} power levels of width {bin_width:g}; {quota} spawns each '
              f'(~{n_levels * quota} samples, target {n_samples})')

    # Pick an equal quota from each level: unique points first, duplicates only
    # once a level has no unused possible points left.
    spawn_x, spawn_y = [], []
    duplicated_levels = 0
    for lv in levels:
        sel = np.where(level_idx == lv)[0]
        if sel.size >= quota:
            pick = np.random.choice(sel, size=quota, replace=False)
        else:
            # All possible points on this level are used; duplicate to reach quota
            duplicated_levels += 1
            extra = np.random.choice(sel, size=quota - sel.size, replace=True)
            pick = np.concatenate([sel, extra])
        spawn_x.append(uniq_pos[pick, 0])
        spawn_y.append(uniq_pos[pick, 1])

    spawn_x = np.concatenate(spawn_x)
    spawn_y = np.concatenate(spawn_y)

    # Interweave levels so the training set isn't ordered by power
    order = np.random.permutation(spawn_x.size)
    spawn_x = spawn_x[order]
    spawn_y = spawn_y[order]
    n_total = spawn_x.size

    if verbose and duplicated_levels:
        print(f'{duplicated_levels} level(s) had fewer possible points than the '
              f'quota of {quota}; duplicated spawns to keep counts equal')

    # -------------------------------- Build the model rows from each spawn --------------------------------

    # The actual beam peak in this scan. Every sample's power readings are
    # rescaled so its effective max power is a random value in [0.5, 5]. This
    # stops the model from keying off an absolute power level and lets it work
    # with almost any peak power at deployment time.
    peak_power = float(power[peak])

    mod_counts = np.zeros((n_total, 1 + moves * 3))  # Last data measurements and current power
    corrections = np.zeros((n_total, 2))             # Positive or negative direction

    for i in tqdm(range(n_total)):  # NOTE: the center of the beam is at (0, 0)

        x = int(spawn_x[i])
        y = int(spawn_y[i])

        # Per-sample random max power: scale all readings so this scan's peak
        # maps to a value drawn uniformly from [0.5, 5].
        power_scale = random.uniform(0.5, 5.0) / peak_power

        current = get_power(x, y) * power_scale

        axis = []
        direction = []
        pwr = []

        xp = x
        yp = y


        for j in range(moves):  # create random last moves
            if j == 0:
                temp_axis = random.randint(0, 1)
            else:
                if temp_axis == 0:
                    temp_axis = 1
                else:
                    temp_axis = 0

            temp_dir = random.choice([-1, 1])

            steps = random.randint(0,100)

            # Step backwards to find where we were before
            if temp_axis == 0:
                xp -= temp_dir * steps
            else:
                yp -= temp_dir * steps

            axis.append(temp_axis)
            direction.append(temp_dir)
            pwr.append(get_power(xp, yp) * power_scale)  # power at that earlier position

        row = [current]                          # current power state
        for j in range(moves):                   # (j+1) moves ago
            row.extend([axis[j], direction[j], pwr[j]])
        mod_counts[i] = row

        # clockwise

        # Last move
        if axis[0] == 0: # This move is Y
            steps = abs(y)/4
        else: # This move is X
            steps = abs(x)/4
        corrections[i][1] = steps

        # Direction toward center (0,0) along THIS move's axis.
        coord = y if axis[0] == 0 else x
        if coord < 0:   # need to move + to reach center
            corrections[i][0] = 1
        elif coord > 0: # need to move - to reach center
            corrections[i][0] = -1


    return mod_counts, corrections
