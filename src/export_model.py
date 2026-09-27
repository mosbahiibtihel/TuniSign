import tensorflow as tf

print("TensorFlow:", tf.__version__)

model = tf.keras.models.load_model(
    "../models/tunisign_model.keras"
)

print("Model loaded successfully!")

model.export("../models/tunisign_saved_model")

print("SavedModel exported successfully!")