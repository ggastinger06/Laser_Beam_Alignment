# Must use Python 3.11 because tensorflow isn't included otherwise
# This code was found in the examples section of moku neural network https://apis.liquidinstruments.com/mnn/examples/Classification.html
# import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from moku.nn import LinnModel, save_linn
# set the seed for repeatability
np.random.seed(42)


# length of the input signal
N = 100

# time series over which to define the signals
T = np.array([t for t in range(N)])

# a range of phases and frequencies to generate signals over
P = [2*p for p in range(N//2)]
O = np.pi/np.array([o for o in range(8,64)])

# a range of widths for our signal anomaly
dT =np.array([10+dt for dt in range(22)])

# create all the signals
signals = []
for o, p in list(itertools.product(O,P)):
    s = np.sin(o*(T+p))
    signals.append(s)

# view some of the signals so we get a sense of what we're generating
plt.figure(figsize=(15,5))
for i in range(0, len(signals), 1000):
    plt.plot(signals[i])

plt.xlabel('Sample points')
plt.ylabel('Amplitude (arb.)')
plt.show()


# list to hold all the signals
sigs_dfct = []
for i in range(2):
    # make a copy so we don't modify the original and then shuffle it
	sig1 = np.copy(signals)
	np.random.shuffle(sig1)

    # for each signal in the list, modify it to have an anomaly and noise
	for s in sig1:
        # choose a starting index and width
		start_idx = np.random.choice(T, replace=True)
		dt = np.random.choice(dT, replace=True)

        # clip to the bounds of the array
		if start_idx + dt >= N:
			start_idx = N-dt

        # create an array of indicies that we will modify
		idxs = np.array([start_idx+d for d in range(dt)])
		idxs = idxs[idxs<N]

        # set all the indices to the same value as the anomaly
		s[idxs] = s[start_idx]

        # flip the symmetry of the anomaly on the second iteration
		if i == 1:
			s[idxs] = s[idxs[-1]]

        # create some noise and add it to the signal
		noise = np.random.rand(N)*0.2
		sigs_dfct.append(s+noise)

# create some non-defective signals with noise
sigs_non = []
for i in range(2):
    # copy the signals to avoid modification
	sig1 = np.copy(signals)
	np.random.shuffle(sig1)

    # add the noise and store them
	for s in sig1:
		noise = np.random.rand(N)*0.2
		sigs_non.append(s+noise)

# plot an example of what we just created
fig, ax = plt.subplots(1, 2, figsize=(15,5), sharey=True)
ax[0].plot(sigs_dfct[-1])
ax[0].plot(range(idxs[0], idxs[-1]), sigs_dfct[-1][idxs[0]:idxs[-1]], 'r')
ax[0].legend(['Signal', 'Anomaly'])
ax[0].set_xlabel('Sample points')
ax[0].set_ylabel('Amplitude (arb.)')
ax[0].set_title('Defective signal')

plt.plot(sigs_non[0])
ax[1].legend(['Signal'])
ax[1].set_xlabel('Sample points')
ax[1].set_title('Non-defective signal')
plt.tight_layout()
plt.show()


# construct the arrays, we will use half of them 
X = np.concatenate([sigs_non[:len(sigs_non) // 2], sigs_dfct[:len(sigs_non) // 2]], axis=0)
y = np.concatenate([np.zeros(len(sigs_non) // 2), np.ones(len(sigs_non) // 2)], axis=0)

# shuffle the arrays to avoid training bias
random_idx = [i for i in range(len(y))]
np.random.shuffle(random_idx)
X = X[random_idx]
y = y[random_idx].reshape(-1, 1)

# get 10% of the random indices 
data_indices = np.arange(0, len(X), 1)
np.random.shuffle(data_indices)
val_length = int(len(X)*0.1) 
train_indices = data_indices[val_length:]
val_indices = data_indices[:val_length]

# separate the training and validation sets
train_X = X[train_indices]
train_y = y[train_indices]
val_X = X[val_indices]
val_y = y[val_indices]


# instantiate the quantised model and set the training data
quant_mod = LinnModel()
quant_mod.set_training_data(training_inputs=train_X, training_outputs=train_y)

# define the model architecture and construct it.
model_definition = [(100, 'relu'), (64, 'relu'), (64, 'relu'), (32, 'relu'), (1,'tanh')]
quant_mod.construct_model(model_definition, show_summary=True, loss='mse')


# Kick off the training
es_config = {'monitor': 'loss', 'patience': 100, 'restore': True}
history = quant_mod.fit_model(epochs=1000, validation_data=(val_X, val_y), es_config=es_config)


# Plot the relevant losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()


def get_accuracy(y_true, y_pred):
    # round up/down to get a binary prediction. This could be modified with different thresholds
    rounded_y_pred = np.round(y_pred)

    # find the accuracy py finding when the difference is 0
    diff = (y_true - rounded_y_pred)
    return np.array(diff==0, dtype=int).sum() / len(diff)

# get the training predictions
preds = quant_mod.predict(train_X)
y_train_acc = get_accuracy(train_y,preds)*100

# get the validation predictions
preds_val = quant_mod.predict(val_X)
y_val_acc = get_accuracy(val_y, preds_val)*100

print('-'*10 + ' Model Accuracy ' + '-'*10 + '\nTrain: \t %.2f %%' % y_train_acc + '\nVal: \t %.2f %%\n' % y_val_acc + '-'*36)

# plot a histogram of all the predictions
plt.hist(preds, bins=np.linspace(0,1,101))
plt.hist(preds_val, bins=np.linspace(0,1,101))
plt.ylabel('Counts')
plt.xlabel('Output value')
plt.show()

# Save the plot
# save_linn(quant_mod, input_channels=1, output_channels=1, file_name='classifier.linn')
