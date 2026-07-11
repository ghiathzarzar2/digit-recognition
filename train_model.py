# train_model.py
# Step 1: Train a CNN to recognize handwritten digits using the MNIST dataset

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. LOAD THE DATA
# MNIST is built into Keras: 60,000 training images + 10,000 test images
# Each image is 28x28 pixels, grayscale, showing a digit 0-9
# ---------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training images: {x_train.shape}")  # (60000, 28, 28)
print(f"Test images: {x_test.shape}")        # (10000, 28, 28)

# ---------------------------------------------------
# 2. PREPROCESS THE DATA
# - Normalize pixel values from [0, 255] to [0, 1] -> helps the model learn faster
# - Reshape to add a "channel" dimension (CNNs expect: height, width, channels)
# ---------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ---------------------------------------------------
# 3. BUILD THE CNN MODEL
# This is the "architecture" of the brain - layers stacked on top of each other
# ---------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),  # helps prevent overfitting
    layers.Dense(10, activation="softmax")  # 10 outputs: one per digit (0-9)
])

model.summary()

# ---------------------------------------------------
# 4. COMPILE THE MODEL
# Tell it how to learn: optimizer, loss function, and what to measure
# ---------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------
# 5. TRAIN THE MODEL
# epochs = how many times it goes through the entire training set
# ---------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1  # use 10% of training data to check progress
)

# ---------------------------------------------------
# 6. EVALUATE ON TEST DATA (data the model has NEVER seen)
# ---------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")  # Expect ~99% with this architecture

# ---------------------------------------------------
# 7. SAVE THE TRAINED MODEL
# This file is the "brain" your backend will load later
# ---------------------------------------------------
model.save("digit_model.h5")
print("Model saved as digit_model.h5")

# ---------------------------------------------------
# 8. (Optional) Plot training history - useful for your report
# ---------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.legend()
plt.title("Accuracy over training")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.title("Loss over training")

plt.savefig("training_history.png")
print("Training history chart saved as training_history.png")
