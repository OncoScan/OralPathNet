"""
CNN Model to estimate with minimal hidden layers.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Function to generate synthetic images
def generate_synthetic_images(num_images, img_size=(224, 224, 3)):
    synthetic_images = np.random.rand(num_images, *img_size) * 255  # Random noise images
    synthetic_labels = np.random.randint(0, 2, size=(num_images,))  # Random binary labels
    return synthetic_images.astype(np.uint8), synthetic_labels

# Generate synthetic data
X_synthetic, y_synthetic = generate_synthetic_images(1000)


# Build the CNN model
def build_cnn_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Change to 'softmax' if multi-class
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',  # Change to 'categorical_crossentropy' if multi-class
                  metrics=['accuracy'])
    return model

# Create the CNN model
cnn_model = build_cnn_model()

# Train the model on synthetic data
cnn_model.fit(X_synthetic, y_synthetic, epochs=10, batch_size=32, validation_split=0.2)
