
"""# Disucssion notes as on 20/09/24

- [ ] Try more epochs -> check for improving accuracy
- [ ] Try more layers if it can help
- [ ] Try GAN for image data (prefer this)
- [ ] Try model with real dataset

GANs can learn image themself, so get better accuracy.

Also model is fine as is, but with synthetic data it should be near perfect (> 90%).
Try GAN -> then move to training real dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
import cv2

# Generator model
def build_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[224, 224, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# GAN class to combine generator and discriminator
class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Generate random noise
        random_latent_vectors = tf.random.normal(shape=(batch_size, 100))

        # Generate fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine real and fake images
        combined_images = tf.concat([real_images, generated_images], axis=0)

        # Labels for real (1) and fake (0) images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Add noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            disc_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Generate random noise
        random_latent_vectors = tf.random.normal(shape=(batch_size, 100))

        # Labels that say "these are real images"
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            gen_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": disc_loss, "g_loss": gen_loss}

# Prepare directories
output_folder = './gan_synthetic_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Instantiate the GAN components
generator = build_generator()
discriminator = build_discriminator()
gan = GAN(generator, discriminator)

# Compile the GAN
gan.compile(
    gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
)

# Training function to generate synthetic images
def train_gan(dataset, epochs=1000, batch_size=32):
    for epoch in tqdm(range(epochs)):
        for real_images in dataset:
            gan.train_step(real_images)

        # Save generated images after each epoch
        random_latent_vectors = tf.random.normal(shape=(1, 100))
        generated_images = generator(random_latent_vectors)
        generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint32)

        # Save the generated image
        img_name = f"gan_image_epoch_{epoch}.jpg"
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, generated_images[0])

# Load dataset (your real images)
def load_dataset(image_folder, batch_size):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_folder,
        image_size=(224, 224),
        batch_size=batch_size
    )
    return dataset

# Load real images from a directory and start GAN training

real_images_dataset = load_dataset('/content/drive/MyDrive/img_data/histo/first_set/', batch_size=32,)

type(real_images_dataset)

# preprocess the dataset to convert the int32 spec to required tf.float32 spec
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])  # Resize to 128x128
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return image, label

#train_dataset = (real_images_dataset
#                 .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#                 .shuffle(buffer_size=1024)
#                 .batch(32, drop_remainder=True)  # Drop last incomplete batch
#                 .prefetch(tf.data.AUTOTUNE))


# Apply preprocessing
real_images_dataset = real_images_dataset.map(preprocess).batch(32, drop_remainder=True)

train_gan(real_images_dataset, epochs=10)

print(real_images_dataset)

train_dataset = (real_images_dataset
                 .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                 .shuffle(buffer_size=1024)
                 .batch(32, drop_remainder=True)  # Drop last incomplete batch
                 .prefetch(tf.data.AUTOTUNE))

train_gan(train_dataset, epochs=10)
