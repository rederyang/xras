import tensorflow as tf

def prepare(ds, batch_size, shuffle=False, augment=False, transform, autotune):
    if shuffle:
        ds = ds.shuffle(1000)
    
    ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(lambda x, y: (transform(x, training=True), y),
                    num_parallel_calls=autotune)
    
    return ds.prefetch(buffer_size=autotune)