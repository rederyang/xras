import tensorflow as tf
from tensorflow import keras

def prepare(ds, batch_size, transform, shuffle=False):
    if shuffle:
        ds = ds.shuffle(len(ds)) # a large enough buffer size is required 
    
    ds = ds.batch(batch_size)

    if transform:
        ds = ds.map(lambda x, y: (transform(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train.astype('float32')
    x_test.astype('float32')
    x_train = x_train / 255.
    x_test = x_test / 255.

    x_train = (x_train - [0.4914, 0.4822, 0.4465]) / [0.247, 0.243, 0.261]
    x_test = (x_test - [0.4914, 0.4822, 0.4465]) / [0.247, 0.243, 0.261]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return ds_train, ds_test