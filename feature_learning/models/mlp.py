
import tensorflow as tf

def get_MLP():
    return [
        tf.keras.layers.GlobalMaxPool2D(data_format='channels_last'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization()
    ]
