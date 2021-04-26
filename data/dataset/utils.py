import tensorflow as tf
from tensorflow import keras

def prepare(ds, batch_size, transform=None, shuffle=False):
    if shuffle:
        ds = ds.shuffle(len(ds)) # a large enough buffer size is required 
    
    ds = ds.batch(batch_size)

    if transform:
        ds = ds.map(lambda x, y: (transform(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)