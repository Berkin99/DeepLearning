import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, utils
from tensorflow.keras.optimizers import Adam

# Dataset path
datasetPath = "Homework1/src/Part2/dataset"

batchSize = 32
imgSize = (128, 128)

# Training data generator with validation split
trainDataset = utils.image_dataset_from_directory(
    datasetPath,
    labels="inferred",  # Automatically infer labels from directory names
    label_mode="int",   # Labels will be integers (for sparse_categorical_crossentropy)
    color_mode="grayscale",  # Convert images to grayscale
    batch_size=batchSize,
    image_size=imgSize,
    shuffle=True,
    seed=123,
    validation_split=0.2,  # Split the data into training (80%) and validation (20%)
    subset="training"  # This will use the training subset
)

trainDataset.repeat()

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
    subset="validation"  # This will use the validation subset
)

validationDataset.repeat()

# Calculate the number of steps per epoch based on the dataset size
steps_per_epoch = int(trainDataset.cardinality().numpy() // batchSize)
validation_steps = int(validationDataset.cardinality().numpy() // batchSize)

# AlexNet Model
model = models.Sequential()

# Input layer: 128x128 grayscale images (1 channel)
model.add(layers.InputLayer(input_shape=(128, 128, 1)))

# Conv1: 11x11 kernel, 96 filters
model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

# Conv2: 5x5 kernel, 256 filters
model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

# Conv3: 3x3 kernel, 384 filters
model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))

# Conv4: 3x3 kernel, 384 filters
model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))

# Conv5: 3x3 kernel, 256 filters
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

# Flatten layer
model.add(layers.Flatten())

# Fully connected layers (FC)
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

# Output layer: 8 classes (assuming 8 shape classes)
model.add(layers.Dense(8, activation='softmax'))  # Softmax for multi-class classification

# Set the learning rate
learning_rate = 0.0002

# Compile model
model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.save('./model.h5')
# Model summary to check the architecture
model.summary()

# Training the model with updated steps per epoch
history = model.fit(
    trainDataset,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    validation_data=validationDataset,
    validation_steps=validation_steps
)

# Plotting performance: Accuracy and Loss
# Plot accuracy
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