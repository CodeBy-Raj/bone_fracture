import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

DATASET_PATH = "dataset/"  # Path to the dataset
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
IMG_SIZE = (224, 224)  # ResNet-50 requires 224x224 images
BATCH_SIZE = 32  # Adjust batch size based on system memory


train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for validation

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)


sample_images, sample_labels = next(train_generator)

plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(sample_images[i])
    plt.axis("off")
plt.show()
  