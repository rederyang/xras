import tensorflow as tf

def deploy():
    return tf.distribute.MirroredStrategy()