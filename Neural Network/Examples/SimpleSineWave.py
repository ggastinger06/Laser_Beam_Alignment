# Must use Python 3.11 because tensorflow isn't included otherwise
# This code was found in the examples section of moku neural network https://apis.liquidinstruments.com/mnn/examples/Simple_sine.html 
# import the relevant libraries for this example
import numpy as np
import matplotlib.pyplot as plt

from moku.nn import LinnModel, save_linn
# set the seed for repeatability
np.random.seed(7)



# generate some very basic training data
X = np.arange(0, np.pi*2, 0.01)
Y = np.cos(X*5)

plt.plot(X, Y, '.')
plt.xticks(np.linspace(0, np.pi*2, 11), labels=["%.1f" % i for i in np.linspace(-1, 1, 11)])
plt.xlabel('Scaled input voltage (arb.)')
plt.ylabel('Scaled output voltage (arb.)')
plt.show()

# get 10% of the random indices 
data_indices = np.arange(0, len(X), 1)
np.random.shuffle(data_indices)
val_length = int(len(X)*0.1) 
train_indices = data_indices[val_length:]
val_indices = data_indices[:val_length]

# separate the training and validation sets
train_X = X[train_indices]
train_Y = Y[train_indices]
val_X = X[val_indices]
val_Y = Y[val_indices]


# create the quantised model object
quant_mod = LinnModel()
quant_mod.set_training_data(training_inputs=train_X.reshape(-1,1), training_outputs=train_Y.reshape(-1,1))

# model definition
model_definition = [(32, 'relu'), (32, 'relu'), (32, 'tanh'), (1, 'linear')]

# build the model
quant_mod.construct_model(model_definition, show_summary=True)

# fit the model
history = quant_mod.fit_model(epochs=1000, validation_data=(val_X.reshape(-1,1), val_Y.reshape(-1,1)))

# plot the losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.xlabel('Epochs')
plt.show()

# use the model to predict the output
preds = quant_mod.predict(X)

# plot the training, validation and model predictions
plt.plot(train_X, train_Y, '.')
plt.plot(val_X, val_Y, 'r.')
plt.plot(X, preds)
plt.xlabel('Scaled input voltage (arb.)')
plt.ylabel('Scaled output voltage (arb.)')
plt.show()

# This will export our trained model to a json file that can be read in by the Moku neural network instrument.
# save_linn(quant_mod, input_channels=1, output_channels=1, file_name='simple_sine_model.linn')
