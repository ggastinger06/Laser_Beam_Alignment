import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf128
import random

from moku.nn import LinnModel
from XYMoveSamples import generate_samples

# Amount of random cursors to simulate with
n_samples = 5000
step = 25
moves = 4 # Length of the move history

# Seed both RNGs so the training data is reproducible. Must match the SEED in PhysicalAlignment.py.
SEED = 7
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------------Generate/import training data here----------------------------------------------------------

# Training data: equal numbers of simulated situations from each listed scan.
# Add or remove scans freely; n_samples is split evenly across them.
# Must match TRAIN_SCANS in PhysicalAlignment.py so the model scaling agrees.
TRAIN_SCANS = [
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris1.1.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris1.2.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris2.0.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris2.1.npz',
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_att.npz',
]
mod_counts, corrections = generate_samples(TRAIN_SCANS, n_samples, moves, step)

# -----------------------------------------------------------TRAINING--------------------------------------------------------
#optimizer = tf128.keras.optimizers.Adam(learning_rate=.005)
laser_mod = LinnModel()
laser_mod.set_training_data(training_inputs=mod_counts, training_outputs=corrections)
out_dim = corrections.shape[1]
model_definition = [(16, 'softsign'), (16, 'softsign'), (out_dim, 'linear')]
laser_mod.construct_model(model_definition, show_summary=True, metrics=['mae'])
history = laser_mod.fit_model(epochs=500, es_config={'patience':25, 'restore':True}, validation_split=0.1)

# Average difference between val loss and loss at the end
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
gap = val_loss[-25:] - loss[-25:]
avg_gap = np.mean(gap)
min_loss = min(loss)

print(f'Minimum loss: {min_loss:.4f}')
print(f'Average (val_loss - loss): {avg_gap:.4f}')

# Same for MAE: lowest MAE and the average val_mae - mae gap
mae = np.array(history.history['mae'])
val_mae = np.array(history.history['val_mae'])
mae_gap = val_mae[-25:] - mae[-25:]
avg_mae_gap = np.mean(mae_gap)
min_mae = min(mae)

print(f'Minimum MAE: {min_mae:.4f}')
print(f'Average (val_mae - mae): {avg_mae_gap:.4f}')

# plot the loss and validation loss as a function of epoch to see how our training went
plt.semilogy(loss)
plt.semilogy(mae)
plt.plot(val_loss)
plt.plot(val_mae)
plt.legend(['loss', 'mae', 'val_loss', 'val_mae'])
plt.xlabel('Epochs')
plt.savefig('loss_and_mae.png', dpi=150, bbox_inches='tight')
plt.show()

# ----------------------------------------Second set of data to test on------------------------------------------

# Test on a different beam scan to gauge generalization
mod_counts, corrections = generate_samples(
    r'C:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust\scan_data_iris1.0.npz',
    n_samples, moves, step,
)

preds = laser_mod.predict(mod_counts, scale=True, unscale_output=True) # The golden line

# -----------------------------------------------------CLAUDE ASSISTED GRAPHING-------------------------------------------------

# Both generators encode one of four directions (+x, -x, +y, -y); they just
# package it differently, so decode each into the same 0-3 labels:
#   2 columns -> ThreeMoveSamples: the [x, y] unit vector gives the direction directly.
#   1 column  -> XYMoveSamples:    a signed scalar along the axis perpendicular to the
#                                  last move; that axis is axis[0], stored in mod_counts[:, 1].
preds = np.asarray(preds)
corrections = np.asarray(corrections)

direction_names = ['+x', '-x', '+y', '-y']

def vec_to_label(vec):
    # Unit vector [x, y] -> direction label 0-3.
    if abs(vec[0]) > abs(vec[1]):
        return 0 if vec[0] > 0 else 1   # +x or -x
    else:
        return 2 if vec[1] > 0 else 3   # +y or -y

def scalar_to_label(scalar, last_axis):
    # Signed scalar along the axis perpendicular to the most recent move.
    if last_axis == 0:                  # last move was X -> correction is along Y
        return 2 if scalar > 0 else 3   # +y or -y
    else:                               # last move was Y -> correction is along X
        return 0 if scalar > 0 else 1   # +x or -x

# This script trains on generate_samples (XYMoveSamples), whose correction is
# [direction_sign, step_size]: column 0 is the signed move along the axis
# perpendicular to the last move (axis[0], stored in mod_counts[:, 1]), and
# column 1 is the step size. Decode the *direction* from the sign in column 0 via
# scalar_to_label. (vec_to_label only applies to ThreeMoveSamples, whose
# correction is a genuine [x, y] unit vector -- feeding it [sign, step_size]
# makes every sample read as +y, since step_size dominates and is never negative.)
last_axes = mod_counts[:, 1].astype(int)   # axis[0]: the most recent move
pred_labels = np.array([scalar_to_label(p[0], a) for p, a in zip(preds, last_axes)])
true_labels = np.array([scalar_to_label(c[0], a) for c, a in zip(corrections, last_axes)])

accuracy = np.mean(pred_labels == true_labels) * 100

n_classes = len(direction_names)

# Build confusion matrix manually
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
for i in range(n_classes):
    for j in range(n_classes):
        axes[0].text(j, i, conf[i, j], ha='center', va='center', fontsize=12)
fig.colorbar(im, ax=axes[0])

# Right: per-direction accuracy bar chart
per_dir = [conf[i, i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
           for i in range(n_classes)]
colors = ['green' if a >= 50 else 'red' for a in per_dir]
axes[1].bar(direction_names, per_dir, color=colors)
chance = 100 / n_classes
axes[1].axhline(chance, color='gray', linestyle='--', label=f'Random chance ({chance:.0f}%)')
axes[1].set_ylim(0, 100)
axes[1].set_xlabel('Direction')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy per Direction')
axes[1].legend()

# Right: predicted vs. true step size (the move magnitude, column 1 of the
# correction). With thousands of samples spread across many spawn positions a
# raw scatter saturates into a blob, so show the point density (hexbin) and
# overlay the binned median prediction with a 10-90% band: the median tracking
# the dashed line means the model recovers the magnitude, and the band width
# shows how consistent it is at each true step size.
true_steps = corrections[:, 1]
pred_steps = preds[:, 1]
step_mae = np.mean(np.abs(pred_steps - true_steps))

hi = max(true_steps.max(), pred_steps.max())
hb = axes[2].hexbin(true_steps, pred_steps, gridsize=40, cmap='Blues',
                    extent=(0, hi, 0, hi), mincnt=1)
fig.colorbar(hb, ax=axes[2], label='Samples')

n_bins = 20
bin_edges = np.linspace(0, true_steps.max(), n_bins + 1)
bin_idx = np.clip(np.digitize(true_steps, bin_edges) - 1, 0, n_bins - 1)
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
axes[2].plot(centers, med, color='green', label='Median prediction')

axes[2].plot([0, hi], [0, hi], 'k--', label='Perfect prediction')
axes[2].set_xlim(0, hi)
axes[2].set_ylim(0, hi)
axes[2].set_xlabel('True step size')
axes[2].set_ylabel('Predicted step size')
axes[2].set_title(f'Step Size  —  MAE: {step_mae:.2f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('direction_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Overall accuracy: {accuracy:.1f}%')
for i, name in enumerate(direction_names):
    print(f'  {name}: {per_dir[i]:.1f}%')
print(f'Step-size MAE: {step_mae:.2f}')


#laser_mod.model.save_weights('laser.weights.h5')
#print('Weights saved')