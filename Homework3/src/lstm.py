import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class Data1D:
    def __init__(self, name:str, data:list):
        self.name = name
        self.data = data
        self.max_x = max(data)
        self.min_x = min(data)

    def normalized(self)->list:
        return [(x - self.min_x) / (self.max_x - self.min_x) for x in self.data]

    def denormalize(self, new:list)->list:
        return [x * (self.max_x - self.min_x) + self.min_x for x in new]

# Extract the data
dates = []
tsla_open = []
tsla_high = []
tsla_low = []
tsla_close = []
tsla_adj = []
tsla_volume = []

with open("Homework3 - TSLA.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        dates.append(row[0])
        tsla_open.append(float(row[1]))
        tsla_high.append(float(row[2]))
        tsla_low.append(float(row[3]))
        tsla_close.append(float(row[4]))
        tsla_adj.append(float(row[5]))
        tsla_volume.append(float(row[6]))

tsla = {
    "Open"   : Data1D("Open", tsla_open),
    "Close"  : Data1D("Close", tsla_close),
    "High"   : Data1D("High", tsla_high),
    "Low"    : Data1D("Low", tsla_low),
    "Adj"    : Data1D("Adj", tsla_adj),
    "Volume" : Data1D("Volume", tsla_volume)
}

# Normalized
open_norm = Data1D("Open", tsla_open).normalized()
high_norm = Data1D("High", tsla_high).normalized()
low_norm = Data1D("Low", tsla_low).normalized()
close_norm = Data1D("Close", tsla_close).normalized()

data = np.array([open_norm, high_norm, low_norm, close_norm]).T

def create_multivariate_sequences(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step, -1])  # Network output : Close values
    return np.array(X), np.array(y)

# Time interval
time_step = 100

# Sequenced data
X, y = create_multivariate_sequences(data, time_step)

# Training & Validation sets (%70 train, %30 validate)
split_index = int(0.7 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(64),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1) 
])

# Compile
model.compile(optimizer="adam", loss="mean_squared_error")

# Model Train
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=20, 
    batch_size=32,
)

# Predict
predictions = model.predict(X_val)

# Denormalized predictions
denormalized_predictions = Data1D("Close", tsla_close).denormalize(predictions.flatten())
denormalized_actual = Data1D("Close", tsla_close).denormalize(y_val)

# Extract loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", marker='o')
plt.plot(val_loss, label="Validation Loss", marker='o')
plt.title("Model Training and Validation Loss Over Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Plot the denormalized predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(denormalized_actual, label="Actual Close Price", color='blue', alpha=0.7)
plt.plot(denormalized_predictions, label="Predicted Close Price", color='red', alpha=0.7)
plt.title("Actual vs Predicted Close Prices (Validation Set)", fontsize=16)
plt.xlabel("Time Steps", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

