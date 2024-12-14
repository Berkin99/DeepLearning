import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Interactive testing loop
while True:
    ui = input("Type 'exit' to quit or press Enter to test a random image: ")
    if ui.lower() == "exit":
        break

    # Select a random index
    xrand = random.randint(0, len(x_test) - 1)
    xn = x_test[xrand].reshape(1, 28, 28, 1)  # Add batch dimension for prediction
    predicted = model.predict(xn)
    
    # Display the image
    plt.imshow(x_test[xrand], cmap='gray')
    plt.title(f"Model Prediction: {predicted.argmax()}")
    plt.show()
