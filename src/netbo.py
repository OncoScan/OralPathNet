"""# Efficient NetB0"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import os
import numpy as np
import cv2

# Create directories for synthetic data
os.makedirs('data/train/class1', exist_ok=True)
os.makedirs('data/train/class2', exist_ok=True)
os.makedirs('data/validation/class1', exist_ok=True)
os.makedirs('data/validation/class2', exist_ok=True)
os.makedirs('data/test/class1', exist_ok=True)
os.makedirs('data/test/class2', exist_ok=True)

# Function to create synthetic images
def create_synthetic_image(class_name, set_name, count):
    for i in range(count):
        image = np.random.rand(224, 224, 3) * 255
        image = image.astype(np.uint8)
        cv2.circle(image, (112, 112), 50, (0, 255, 0), -1)  # Add a synthetic feature
        cv2.imwrite(f'data/{set_name}/{class_name}/img_{i}.jpg', image)

# Generate synthetic images
for cls in ['class1', 'class2']:
    for set_name in ['train', 'validation', 'test']:
        create_synthetic_image(cls, set_name, 100)  # 100 images per class per set

# Load and preprocess the dataset
# Assuming dataset is organized into 'train', 'validation', and 'test' directories
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Change to 'categorical' if you have more than 2 classes
)

validation_generator = val_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Build and compile the model
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(1, activation='sigmoid')  # Change to 'softmax' with more classes
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Callbacks for optimization
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6)

"""  Note:
- model.fit might get error: try with x,y arguments (78846949 Stackoverflow)

"""

# Train the model
history = model.fit(
    train_generator, # ERROR FIXME
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=2,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Load the best model and use it for prediction
best_model = models.load_model('best_model.keras')
predictions = best_model.predict(test_generator)

model.save('our_model.keras')

models.load_model('/content/our_model.keras')
