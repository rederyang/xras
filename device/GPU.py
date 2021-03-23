import tensorflow as tf

def deploy_gpu():
    return tf.distribute.MirroredStrategy()