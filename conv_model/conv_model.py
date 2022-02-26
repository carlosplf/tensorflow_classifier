import tensorflow as tf
from tensorflow import keras
from keras import layers


class ConvModel():
    def __init__(self):
        """
        Create a Sequential NN Model using Tensorflow.
        The model uses 2D Convolutional processing and Max Pooling 2D to        process the images.
        """
        num_classes = 6
        model = keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.model = model

