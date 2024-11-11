import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, utils
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam

# Dataset path
datasetPath = "Homework1/src/Part2/dataset"

batchSize = 32
imgSize = (128, 128)

# Training data generator with validation split
trainDataset = utils.image_dataset_from_directory(
    datasetPath,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batchSize,
    image_size=imgSize,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training"
)

# Validation data generator
validationDataset = utils.image_dataset_from_directory(
    datasetPath,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batchSize,
    image_size=imgSize,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation"
)

# Add normalization (Rescaling) layer to your data pipeline
trainDataset = trainDataset.map(lambda x, y: (x / 255.0, y))  # Normalize to [0, 1]
validationDataset = validationDataset.map(lambda x, y: (x / 255.0, y))  # Normalize to [0, 1]

#  AlexNet Model with reduced complexity
model = models.Sequential()

# Input layer: 128x128 grayscale images (1 channel)
model.add(layers.InputLayer(input_shape=(128, 128, 1)))

# Conv1: 5x5 kernel, 32 filters (Smaller filters and reduced filters)
model.add(layers.Conv2D(32, (5, 5), strides=2, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Conv2: 3x3 kernel, 64 filters (Reduce the number of filters)
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Conv3: 3x3 kernel, 128 filters (Reduce the number of filters)
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

# Flatten layer
model.add(layers.Flatten())

# Fully connected layers (FC)
model.add(layers.Dense(128, activation='relu'))  # Reduced from 4096 to 128
model.add(layers.Dropout(0.5))                   # Dropout layer to avoid overfitting
model.add(layers.Dense(128, activation='relu'))  # Reduced from 4096 to 128
model.add(layers.Dropout(0.5))

# Output layer: 8 classes (assuming 8 shape classes)
model.add(layers.Dense(8, activation='softmax'))  # Softmax for multi-class classification

# Compile model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.save('./model.h5')
# Model summary to check the architecture
model.summary()

# Training the model with updated steps per epoch
history = model.fit(
    trainDataset,
    epochs=100,
    validation_data=validationDataset,
)

# Plotting performance: Accuracy and Loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
