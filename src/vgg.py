"""
VGGmodel

I doubt method is not proper. Got similar result for VGG and resnet.

This should not be simple code of same nature.
"""

# call the modules

from tensorflow.keras.applications import VGG16

# Load VGG16 model pre-trained on ImageNet data
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Create a new model on top of VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Freeze the weights of the VGG16 base model
base_model.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10)

len(train_generator)
