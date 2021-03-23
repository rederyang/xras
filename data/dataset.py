import tensorflow as tf

def ready_dataset(name, normalize=True):

    def normalize(x, normal_params):
        mean = normal_params['mean']
        std = normal_params['std']
        return (x-mean) / std

    if name == 'cifar10':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train.astype('float32')
        y_train.astype('float32')

        if normalize: # normalize
            cifar10_np = {}
            cifar10_np['mean'] = [0.4914, 0.4822, 0.4465]
            cifar10_np['std'] = [0.247, 0.243, 0.261]
            x_train = normalize(x_train, cifar10_np)
            x_test = normalize(x_test, cifar10_np)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes) # to category
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        return ds_train, ds_test