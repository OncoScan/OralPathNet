"""
Our Model = CNN model with 10 pretrained models as layers
"""

"""
Dependency:
pip install albumentations opencv-python
pip install --upgrade albumentations
pip install --upgrade opencv-python
pip show albumentations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    DenseNet121, DenseNet169, DenseNet201,
    MobileNetV2, ResNet50, ResNet101,
    Xception, VGG16, VGG19
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
import albumentations as A
from tqdm import tqdm

def create_synthetic_dataset(output_folder, num_images=1000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define augmentations
    augmentations = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(),
        A.ElasticTransform(),
        A.MotionBlur(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])

    # Create synthetic images
    for i in tqdm(range(num_images)):
        # Start with a blank image or a basic pattern
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply augmentations
        augmented = augmentations(image=image)
        augmented_image = augmented['image']

        # Save the synthetic image
        image_name = f"synthetic_image_{i}.jpg"
        image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(image_path, augmented_image)

# Generate 1000 synthetic images
create_synthetic_dataset('./synthetic_images', num_images=1000)

import os
import random
import shutil

# Set paths
source_dir = './synthetic_images'
train_dir = os.path.join(source_dir, 'train')
validation_dir = os.path.join(source_dir, 'validation')

# Create training and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# List all image files in the source directory
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Shuffle the images randomly
random.shuffle(all_images)

# Calculate split index
split_index = int(0.8 * len(all_images))

# Split images into training and validation sets
train_images = all_images[:split_index]
validation_images = all_images[split_index:]

# Move images to respective directories
for image in train_images:
    shutil.move(os.path.join(source_dir, image), os.path.join(train_dir, image))

for image in validation_images:
    shutil.move(os.path.join(source_dir, image), os.path.join(validation_dir, image))

print(f"Moved {len(train_images)} images to {train_dir}")
print(f"Moved {len(validation_images)} images to {validation_dir}")

from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import DenseNet121, DenseNet169, DenseNet201, MobileNetV2, ResNet50, ResNet101, Xception, VGG16, VGG19, InceptionV3

def rename_layers(model, prefix):
    for layer in model.layers:
        layer._name = f'{prefix}_{layer.name}'

def create_combined_model():
    input_layer = Input(shape=(224, 224, 3))

    # Load the base models with pre-trained weights
    densenet121_base = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(densenet121_base, "densenet121")

    densenet169_base = DenseNet169(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(densenet169_base, "densenet169")

    densenet201_base = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(densenet201_base, "densenet201")

    mobilenetv2_base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(mobilenetv2_base, "mobilenetv2")

    resnet50_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(resnet50_base, "resnet50")

    resnet101_base = ResNet101(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(resnet101_base, "resnet101")

    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(xception_base, "xception")

    vgg16_base = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(vgg16_base, "vgg16")

    vgg19_base = VGG19(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(vgg19_base, "vgg19")

    inceptionv3_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer)
    rename_layers(inceptionv3_base, "inceptionv3")

    # Add custom layers on top of one of the base models (for example, DenseNet121)
    x = densenet121_base.output
    x = GlobalAveragePooling2D(name="gap_layer")(x)
    x = Dense(1024, activation='relu', name="dense_layer_1")(x)
    output_layer = Dense(1, activation='sigmoid', name="output_layer")(x)

    # Create the final model
    model = Model(inputs=input_layer, outputs=output_layer, name="combined_model")

    return model

# Create the combined model
model = create_combined_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Optional: Print layer names to verify uniqueness
for layer in model.layers:
    print(layer.name)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Synthetic dataset generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    './synthetic_images',  # Update with the correct path to synthetic images
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    './synthetic_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Print data generator stats
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")


# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=1000
)

from google.colab import drive
drive.mount('/content/drive')

""" To pull gdrive
mkdir /content/drive/MyDrive/img_data
unzip -d /content/drive/MyDrive/img_data/histo /content/drive/MyDrive/Colab\ Notebooks/ftmp4cvtmb-1.zip
"""

#save the model
model.save('/content/drive/MyDrive/img_data/oc_cnn_model.keras')
print("Model saved as our_model.h5")

model_path = "/content/drive/MyDrive/img_data/oc_cnn_model.keras"

#This cell is to delete the generated synthetic images...
#run only if you want new synthetic images
import shutil

folder_path = '/content/img_data/histo'
#shutil.rmtree(folder_path)
