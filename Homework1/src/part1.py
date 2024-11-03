import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

def function(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    y1 = x1 * x3 + 1.2 * x1 * x5 - x6 * x7 * x8 - 2 * x1**2 * x8 + x5
    y2 = x1 * x5 * x6 - x3 * x4 - 3 * x2 * x3 + 2 * x2**2 * x4 - 2 * x7 * x8 - 1
    y3 = 2 * x3**2 - x5 * x7 - 3 * x1 * x4 * x6 - x1**2 * x2 * x4 - 1
    y4 = -x6**3 + 2 * x1 * x3 * x8 - x1 * x4 * x7 - 2 * x5**2 * x2 * x4 - x8
    y5 = x1**2 * x5 - 3 * x3 * x4 * x8 + x1 * x2 * x4 - 3 * x6 - x1**2 * x7 + 2
    y6 = x1**2 * x3 * x6 - x3 * x5 * x7 + x3 * x4 + 2.2 * x4 + x2**2 * x3 - 2.1
    return [y1, y2, y3, y4, y5, y6]

def dataGenerator(xlist, func, size):
    output = []
    for i in range(size):
        y = func(xlist[i])
        output.append(y)
    return output

def noiseGenerator(xlist, mean, sigma):
    output = []
    for y in xlist:
        noise = np.random.normal(mean, sigma, len(y))
        output.append(y + noise)
    return output

# Model training
# 1. Choose 𝑁t to be 1000.
# 2. In your training data add some noise to 𝑦_i’s from a normal distribution with 𝜇 = 0.0 and 𝜎 = 0.001.

Nt = 1000
Nv = 1000
noiseMean = 0.0
noiseStddev = 0.001

xt = np.random.random((Nt, 8)).tolist() # Training Input array
yt = dataGenerator(xt, function, Nt)    # Training Output array
yt = noiseGenerator(yt, noiseMean, noiseStddev)

xv = np.random.random((Nv, 8)).tolist() # Validation Input array
yv = dataGenerator(xv, function, Nv)    # Validation Output array

# 3. Build a feed forward network with exactly 3 hidden layers:
# * Each layer should include exactly 6 nodes in the beginning.
# * Use a combination of activation functions in these layers (use the same activation for each node at a given layer).
# 4. Define your loss function:
# * Use MSE for loss function.
# 5. Train your algorithm with SGD.
# * Use appropriate learning rates and the number of epochs.
# * Report the training and validation errors.

epochs = 100
learningRate = 0.05
model = Sequential()

model.add(Dense(6, input_shape=(8,), activation='relu')) 
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))

model.add(Dense(6, activation='linear')) # Output layer : 6 neuron, linear

# Compile 
sgd = SGD(learning_rate=learningRate)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

# Train
history = model.fit(
    np.array(xt), 
    np.array(yt), 
    validation_data=(np.array(xv), np.array(yv)),
    epochs=epochs, 
    batch_size=32, 
    verbose=1
)

# Plot
plt.plot(history.history['mse'], label='Training Sample', color ='cyan')
plt.plot(history.history['val_mse'], label='Test Sample', color ='orange')
plt.xlabel('Epochs')
plt.ylabel('Prediction Error MSE')
plt.legend()
plt.show()

# 6. Repeat Steps 2-4 with another set of activation functions (3 different combinations), learning
# rates (3 different schemes) and number of epochs (after finding a reasonable number of
# epochs in the first trial, increase by 50% for 2 times).

# 7. Choose your best parameters after Step 5.