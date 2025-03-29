import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import os

# Load the dataset
from flask_app.app import train_generator, valid_generator  # Import from app.py

# Load pre-trained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers

# Build classification model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # Adjust based on classes
])

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# Train the model
EPOCHS = 10
model.fit(train_generator, epochs=EPOCHS, validation_data=valid_generator)

# Save the model
MODEL_PATH = "flask_app/bone_fracture_model.h5"
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
