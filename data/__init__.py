import tensorflow as tf

def prepare(ds, batch_size, shuffle=False, augment=False, transform):
    if shuffle:
        ds = ds.shuffle(len(ds)) # a large enough buffer size is required 
    
    ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(lambda x, y: (transform(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)