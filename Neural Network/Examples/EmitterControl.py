import numpy as np                     # numerical array operations
import matplotlib.pyplot as plt        # plotting
from tqdm import tqdm                  # wraps loops to show a progress bar
import tensorflow as tf128             # TensorFlow deep learning library, aliased as tf128
                                       # NOTE: unusual alias — normally imported as 'tf'

from moku.nn import LinnModel, save_linn       # proprietary Moku neural network class and save
                                               # function — exact internals uncertain
from MultiAdjust.emitter_simulator import QuantumEmitter   # proprietary quantum emitter simulation class
                                               # exact internals uncertain

# set the seed for repeatability — commented out, so results are not reproducible run-to-run
#np.random.seed(7)
#tf128.random.set_seed(7)

# create 100 evenly spaced points from -10µm to +10µm along one axis
x = np.linspace(-10e-6, 10e-6, 100)

# build a 2D grid of (X, Y) coordinates — indexing='ij' means X varies along rows, Y along columns
X, Y = np.meshgrid(x, x, indexing='ij')

# create a simulated quantum emitter with 780nm wavelength and 5µm beam waist
qe_sim = QuantumEmitter(wavelength=780e-9, waist=5e-6)

# tell the simulator which spatial grid to use
qe_sim.set_XY(X, Y)

# compute the electric field amplitude at every (X,Y) point; 1e-10 is likely a time or z parameter
# — exact meaning of third argument is uncertain without documentation
intensity = np.abs(qe_sim.E(X, Y, 1e-10))**2   # intensity = |E|^2

# plot the beam intensity map as a 2D image
plt.imshow(intensity, extent=[-10, 10, -10, 10])
plt.xlabel('X axis ($\mu$m)')
plt.ylabel('Y axis ($\mu$m)')
plt.colorbar()
plt.show()


def random_walk(step_size, input_array, random_start=False):
    """
    Generates a 1D random walk clipped to [-1, 1].
    step_size:    standard deviation of each random step (controls how fast the walk moves)
    input_array:  only its .size is used — determines number of steps in the walk
    random_start: if True, starts at a random position in [-1,1]; otherwise starts at 0
    """

    if random_start:
        # draw a single uniform random starting value between -1 and 1
        running_value = np.random.uniform(-1, 1, 1)[0]
    else:
        running_value = 0   # start the walk at 0

    # draw one random step per time point from a normal distribution centred at 0
    output_array = np.random.normal(0, step_size, (input_array.size, 1))

    # override the first entry to set the starting position
    output_array[0] = running_value

    for idx in range(output_array.shape[0]):
        if idx != 0:
            # add current random step to previous position (cumulative sum)
            # clip to [-1, 1] so the walk stays within bounds
            output_array[idx] = np.clip(
                output_array[idx] + output_array[idx - 1], -1, 1
            )

    return output_array   # shape: (input_array.size, 1)


# time axis with 1000 points from 0 to 1 — used only to visualise walk behaviour
T = np.linspace(0, 1, 1000)

# plot walks with three different step sizes to show how step_size controls smoothness
steps = [0.1, 0.01, 0.001]
for step_size in steps:
    walk = random_walk(step_size, T)   # generate walk for this step size
    plt.plot(T, walk)

plt.legend(steps)
plt.xlabel('Time (arb.)')
plt.ylabel('Walk position (arb.)')
plt.show()


# longer time axis with 2500 points — used for generating the actual training data
T = np.linspace(0, 1, 2500)

# generate four independent random walks, one for each degree of freedom
X_offset = random_walk(0.1, T)   # beam x-position offset,  shape (2500, 1), range [-1, 1]
Y_offset = random_walk(0.1, T)   # beam y-position offset,  shape (2500, 1), range [-1, 1]
X_angle  = random_walk(0.1, T)   # beam x-angle,            shape (2500, 1), range [-1, 1]
Y_angle  = random_walk(0.1, T)   # beam y-angle,            shape (2500, 1), range [-1, 1]

# scale offsets from [-1, 1] to [-4µm, +4µm] — physical displacement range of the beam
X_offset *= 4e-6
Y_offset *= 4e-6

# scale angles: [-1,1] → [0,2] → [0,10] degrees
# so beam angle varies between 0° and 10° in x and y
X_angle += 1
X_angle *= 5
Y_angle += 1
Y_angle *= 5


# pre-allocate arrays to store simulation results
counts     = np.zeros(T.size)           # base photon counts at each time step, shape (2500,)
mod_counts = np.zeros((T.size, 4))      # modulated counts (4 measurements per step) — likely
                                        # the NN input features; exact format uncertain
corrections = np.zeros((T.size, 4))    # corrections needed at each step — the NN training targets


for i in tqdm(range(T.size)):   # loop over every time step, showing a progress bar

    # extract the scalar offset and angle values for this time step
    x_off = X_offset[i]   # x beam offset in metres
    y_off = Y_offset[i]   # y beam offset in metres
    x_ang = X_angle[i] * np.pi / 180   # x angle converted from degrees to radians
    y_ang = Y_angle[i] * np.pi / 180   # y angle converted from degrees to radians

    # pack offsets into a tuple
    offsets = (x_off, y_off)

    # compute shear values from angles — new_scale() meaning is uncertain without documentation
    # np.pi/2 - angle converts from elevation angle to angle from horizontal
    shears = (
        qe_sim.new_scale(np.pi/2 - x_ang),
        qe_sim.new_scale(np.pi/2 - y_ang)
    )

    # convert angles to the same (pi/2 - angle) convention used by the simulator
    angles = [np.pi/2 - x_ang, np.pi/2 - y_ang]

    # run one simulation time step with these beam parameters
    # returns modulated count measurements (4 values) — exact format uncertain
    all_counts = qe_sim.time_step(offsets, shears, angles)

    # get the base (unmodulated) photon count for this step
    base_count = qe_sim.get_counts()

    # store modulated counts (NN inputs) and base count for this step
    mod_counts[i] = all_counts
    counts[i] = base_count

    # compute the correction vector: how far the current state is from the target [0, 0, 1, 1]
    # target is zero offset (0, 0) and unit shear (1, 1)
    # this is what the NN will learn to predict
    corrections[i] = (
        np.array([0, 0, 1, 1])
        - np.array([*offsets, *shears]).flatten()   # unpack offsets and shears into one array
    )


# instantiate the Moku LINN (Laser Intelligent Neural Network?) model
# — exact class behaviour uncertain without documentation
quant_mod = LinnModel()

# pass the modulated counts as inputs and the required corrections as outputs for training
quant_mod.set_training_data(
    training_inputs=mod_counts,
    training_outputs=corrections
)

# define the 5-layer network architecture as a list of (neurons, activation) tuples
# layers: 100 relu → 100 relu → 64 relu → 64 relu → 4 linear (one output per correction)
model_definition = [
    (100, 'relu'),
    (100, 'relu'),
    (64,  'relu'),
    (64,  'relu'),
    (4,   'linear')   # linear output for regression (predicting continuous correction values)
]

# build the model according to the definition and print a summary of its layers and parameters
quant_mod.construct_model(model_definition, show_summary=True)


# train the model for up to 500 epochs
# early stopping: stop if validation loss doesn't improve for 50 epochs, restore best weights
history = quant_mod.fit_model(
    epochs=500,
    es_config={'patience': 50, 'restore': True},
    validation_split=0.1   # reserve 10% of data for validation
)


# plot training and validation loss on a log scale to assess convergence
plt.semilogy(history.history['loss'])       # training loss per epoch
plt.plot(history.history['val_loss'])       # validation loss per epoch
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()


# run the trained model on the full training set to get predicted corrections
# scale=True and unscale_output=True suggest internal normalisation is applied/reversed
# — exact scaling behaviour uncertain without documentation
preds = quant_mod.predict(mod_counts, scale=True, unscale_output=True)

# pre-allocate array to store photon counts after applying the predicted corrections
counts_cor = np.zeros(T.size)

for i in tqdm(range(T.size)):   # loop over every time step again

    # re-extract beam parameters for this step (same as training loop)
    x_off = X_offset[i]
    y_off = Y_offset[i]
    x_ang = X_angle[i] * np.pi / 180
    y_ang = Y_angle[i] * np.pi / 180

    offsets = (x_off, y_off)
    shears  = (
        qe_sim.new_scale(np.pi/2 - x_ang),
        qe_sim.new_scale(np.pi/2 - y_ang)
    )
    angles = [np.pi/2 - x_ang, np.pi/2 - y_ang]

    # apply the original (uncorrected) disturbance — result discarded with _
    _ = qe_sim.time_step(offsets, shears, angles)

    # apply a second time step using the NN-predicted corrections added to the disturbance
    # preds[i][0] and [1] correct the x and y of