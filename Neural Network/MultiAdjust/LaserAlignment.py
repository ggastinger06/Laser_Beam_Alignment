"""Train the laser-alignment model on simulated beam scans and save its weights.

Run this once to produce laser.weights.h5, which PhysicalAlignment.py then loads
to drive the picomotors. Saves two diagnostic plots: loss_and_mae.png and
direction_accuracy.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random

from moku.nn import LinnModel
from XYMoveSamples import generate_samples

# ============================== USER SETTINGS ==============================
n_samples = 20000   # number of random cursors to simulate
moves = 4           # length of the move history fed to the model

SEED = 7            # seeds both RNGs for reproducibility; MUST match PhysicalAlignment.py
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # this script's folder; scans live alongside it

# Training scans: n_samples is split evenly across them. Add/remove freely, but
# this list MUST match TRAIN_SCANS in PhysicalAlignment.py so the scaling agrees.
TRAIN_SCANS = [
    os.path.join(DATA_DIR, 'scan_data_iris1.1.npz'),
    os.path.join(DATA_DIR, 'scan_data_iris1.2.npz'),
    os.path.join(DATA_DIR, 'scan_data_iris2.0.npz'),
    os.path.join(DATA_DIR, 'scan_data_iris2.1.npz'),
    os.path.join(DATA_DIR, 'scan_data_att.npz'),
]
TEST_SCAN = os.path.join(DATA_DIR, 'scan_data_iris1.0.npz')   # held-out scan, to gauge generalization

EPOCHS = 500
MODEL_DEFINITION = [(32, 'softsign'), (32, 'softsign')]   # output layer is appended below
WEIGHTS_OUT = 'laser.weights.h5' # Name for weights file
# ===========================================================================

np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------- Training data ----------------------------------
inputs, corrections = generate_samples(TRAIN_SCANS, n_samples, moves)

# ---------------------------------- Training ----------------------------------
laser_mod = LinnModel()
laser_mod.set_training_data(training_inputs=inputs, training_outputs=corrections)
out_dim = corrections.shape[1]
model_definition = MODEL_DEFINITION + [(out_dim, 'linear')]
laser_mod.construct_model(model_definition, show_summary=True, metrics=['mae'])
history = laser_mod.fit_model(epochs=EPOCHS, es_config={'patience': 25, 'restore': True},
                              validation_split=0.1)

# Lowest loss and the average val_loss - loss gap over the last 25 epochs
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
avg_gap = np.mean(val_loss[-25:] - loss[-25:])
min_loss = min(loss)
print(f'Minimum loss: {min_loss:.4f}')
print(f'Average (val_loss - loss): {avg_gap:.4f}')

# Same for MAE: lowest MAE and the average val_mae - mae gap
mae = np.array(history.history['mae'])
val_mae = np.array(history.history['val_mae'])
avg_mae_gap = np.mean(val_mae[-25:] - mae[-25:])
min_mae = min(mae)
print(f'Minimum MAE: {min_mae:.4f}')
print(f'Average (val_mae - mae): {avg_mae_gap:.4f}')

# Plot loss/MAE vs epoch to see how training went
plt.semilogy(loss)
plt.semilogy(mae)
plt.plot(val_loss)
plt.plot(val_mae)
plt.legend(['loss', 'mae', 'val_loss', 'val_mae'])
plt.xlabel('Epochs')
plt.savefig('loss_and_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------- Test on a held-out scan ----------------------------------
inputs, corrections = generate_samples(TEST_SCAN, n_samples, moves)
preds = laser_mod.predict(inputs, scale=True, unscale_output=True)

# ---------------------------------- Diagnostic plots ----------------------------------
preds = np.asarray(preds)
corrections = np.asarray(corrections)

direction_names = ['+x', '-x', '+y', '-y']

def scalar_to_label(scalar, last_axis):
    # Signed correction scalar along the axis perpendicular to the most recent move
    if last_axis == 0:                  # last move was X -> correction is along Y
        return 2 if scalar > 0 else 3   # +y or -y
    else:                               # last move was Y -> correction is along X
        return 0 if scalar > 0 else 1   # +x or -x

# Correction is [direction_sign, step_size]; column 0's sign gives the direction.
last_axes = inputs[:, 1].astype(int)   # axis[0]: the most recent move
pred_labels = np.array([scalar_to_label(p[0], a) for p, a in zip(preds, last_axes)])
true_labels = np.array([scalar_to_label(c[0], a) for c, a in zip(corrections, last_axes)])

accuracy = np.mean(pred_labels == true_labels) * 100
n_classes = len(direction_names)

# Build the confusion matrix manually
conf = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(true_labels, pred_labels):
    conf[t][p] += 1

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: confusion matrix
im = axes[0].imshow(conf, cmap='Blues')
axes[0].set_xticks(range(n_classes))
axes[0].set_yticks(range(n_classes))
axes[0].set_xticklabels(direction_names)
axes[0].set_yticklabels(direction_names)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title(f'Confusion Matrix  —  Overall accuracy: {accuracy:.1f}%')
# Annotate each cell with the count and its share of the true row
for i in range(n_classes):
    row_total = conf[i].sum()
    for j in range(n_classes):
        pct = conf[i, j] / row_total * 100 if row_total else 0
        color = 'white' if conf[i, j] > conf.max() / 2 else 'black'
        axes[0].text(j, i, f'{conf[i, j]}\n{pct:.0f}%', ha='center', va='center',
                     fontsize=11, color=color)
fig.colorbar(im, ax=axes[0])

# Middle: per-direction accuracy bar chart
per_dir = [conf[i, i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
           for i in range(n_classes)]
colors = ['green' if a >= 50 else 'red' for a in per_dir]
bars = axes[1].bar(direction_names, per_dir, color=colors)
chance = 100 / n_classes
axes[1].axhline(chance, color='gray', linestyle='--', label=f'Random chance ({chance:.0f}%)')
axes[1].set_ylim(0, 105)
axes[1].set_xlabel('Direction')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy per Direction')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_axisbelow(True)
# Label each bar with its accuracy and test-sample count
for bar, acc, i in zip(bars, per_dir, range(n_classes)):
    axes[1].text(bar.get_x() + bar.get_width() / 2, acc + 1,
                 f'{acc:.0f}%\n(n={conf[i].sum()})', ha='center', va='bottom',
                 fontsize=9)
axes[1].legend()

# Right: chosen step size vs. actual distance from center.
# Target step is a damped fraction of the offset (step = offset / DAMPING), so the
# x-axis is the real distance (correction step * DAMPING). Hexbin shows density;
# the green line/band is the binned median step with a 10-90% spread.
DAMPING = 4
true_center = corrections[:, 1] * DAMPING   # actual steps from center (x)
pred_steps = preds[:, 1]                     # step size the model picks (y)
step_mae = np.mean(np.abs(pred_steps - corrections[:, 1]))

hi_x = true_center.max()
hi_y = max(pred_steps.max(), hi_x / DAMPING)
hb = axes[2].hexbin(true_center, pred_steps, gridsize=40, cmap='Blues',
                    extent=(0, hi_x, 0, hi_y), mincnt=1)
fig.colorbar(hb, ax=axes[2], label='Samples')

n_bins = 20
bin_edges = np.linspace(0, hi_x, n_bins + 1)
bin_idx = np.clip(np.digitize(true_center, bin_edges) - 1, 0, n_bins - 1)
centers, med, p10, p90 = [], [], [], []
for b in range(n_bins):
    in_bin = pred_steps[bin_idx == b]
    if in_bin.size == 0:
        continue
    centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
    med.append(np.median(in_bin))
    p10.append(np.percentile(in_bin, 10))
    p90.append(np.percentile(in_bin, 90))
axes[2].fill_between(centers, p10, p90, color='green', alpha=0.3, label='10–90% band')
axes[2].plot(centers, med, color='green', label='Median chosen step')

axes[2].plot([0, hi_x], [0, hi_x / DAMPING], 'k--', label=f'Perfect (offset / {DAMPING})')
axes[2].set_xlim(0, hi_x)
axes[2].set_ylim(0, hi_y)
axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Actual steps from center')
axes[2].set_ylabel('Predicted step size')
axes[2].set_title(f'Chosen Step vs Distance  —  MAE: {step_mae:.2f} steps')
axes[2].legend()

plt.tight_layout()
plt.savefig('direction_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Overall accuracy: {accuracy:.1f}%')
for i, name in enumerate(direction_names):
    print(f'  {name}: {per_dir[i]:.1f}%')
print(f'Step-size MAE: {step_mae:.2f}')

laser_mod.model.save_weights(WEIGHTS_OUT)
print('Weights saved')
