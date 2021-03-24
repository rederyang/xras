import tensorflow as tf
from tensorflow import keras

def transform(augs):
    return tf.keras.Sequential(augs)