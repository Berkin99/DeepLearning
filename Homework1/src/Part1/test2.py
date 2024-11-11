from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Early stopping with SGDRegressor
model = SGDRegressor(max_iter=1000, tol=1e-3, early_stopping=True, validation_fraction=0.2, random_state=42)
model.fit(X_train, y_train, early_stopping_monitor=('val_loss', 'min'))

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# Define a simple neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stop])