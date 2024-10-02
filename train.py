"""# Real Data Training Arena
"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dir = '/content/drive/MyDrive/img_data/histo/first_set'
test_dir = '/content/drive/MyDrive/img_data/histo/second_set'
img_height, img_width = 224, 224  # Set this to your model's expected input size

built_model = tf.keras.models.load_model(model_path)

train_dataset = image_dataset_from_directory(
    train_dir,
 #   image_size=(img_height, img_width),  # Set your image size
    batch_size=40
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=32
)

"""Dont' use train now! Dataset is to test, and just run on it."""

test_dataset = image_dataset_from_directory(
    test_dir,
    #image_size=(img_height, img_width),  # Use the same size as used in training
    batch_size=32  # Use the same batch size as used in training
)

"""**Note:** Should you normalize the dataset???*italicized text*"""

predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy:.4f}')

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Make predictions
predictions = model.predict(normalized_test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = np.concatenate([y.numpy() for _, y in normalized_test_ds], axis=0)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy:.4f}')

normalized_test_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))

# Make predictions
predictions = model.predict(normalized_test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = np.concatenate([y.numpy() for _, y in normalized_test_ds], axis=0)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy:.4f}')

#normal_test_dir = '/content/drive/MyDrive/img_data/histo/first_set/normal/'
normal_test_dir = "/content/drive/MyDrive/img_data/histo/first_set/"

test_dataset = image_dataset_from_directory(
    normal_test_dir,
    image_size=(img_height, img_width),
    batch_size=32
)
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_test_ds = normal_test_dir.map(lambda x, y: (normalization_layer(x), y))

# Make predictions
predictions = model.predict(normalized_test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = np.concatenate([y.numpy() for _, y in normalized_test_ds], axis=0)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy:.4f}')
