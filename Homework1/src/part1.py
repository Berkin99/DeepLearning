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
Nt = 1000
Nv = 1000
noiseMean = 0.0
noiseStddev = 0.001

xt = np.random.random((Nt, 8)).tolist() # Training Input array
print("xt Element: ", xt[0])

yt = dataGenerator(xt, function, Nt)    # Training Output array
print("yt Element: ", yt[0])

yt = noiseGenerator(yt, noiseMean, noiseStddev)
print("yt Noise Add: ", yt[0])

xv = np.random.random((Nv, 8)).tolist() # Validation Input array
yv = dataGenerator(xv, function, Nv)    # Validation Output array
yv = noiseGenerator(yv, noiseMean, noiseStddev)

# Build the model
model = Sequential()
model.add(Dense(6, input_shape=(8,), activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))

model.add(Dense(6, activation='linear'))

# Compile the model
learningRate = 0.001
sgd = SGD(learning_rate=learningRate)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

# Train the model
epochs = 100
history = model.fit(
    np.array(xt), 
    np.array(yt), 
    validation_data=(np.array(xv), np.array(yv)),
    epochs=epochs, 
    batch_size=32, 
    verbose=1
)

# Evaluate performance
train_loss, train_mse = model.evaluate(np.array(xt), np.array(yt), verbose=0)
val_loss, val_mse = model.evaluate(np.array(xv), np.array(yv), verbose=0)

print(f"Training Loss: {train_loss:.4f}, Training MSE: {train_mse:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation MSE: {val_mse:.4f}")

# Plot training & validation mse values
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
