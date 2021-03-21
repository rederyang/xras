import tensorflow as tf

def setup(device='tpu'):
    if device=='tpu':
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('TPU:', tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
        except:
            print('Failed to connect to TPU.')
            strategy = tf.distribute.get_strategy()
    elif device=='gpu':
        strategy = tf.distribute.MirroredStrategy()

    tf.distribute.experimental_set_strategy(strategy)
    print('Number of replicas:', strategy.num_replicas_in_sync)

