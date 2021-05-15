import tensorflow_datasets as tfds

def load_voc_dataset(large=False):
    if not large:
        ds_train = tfds.load('voc/2007', split='train+validation', shuffle_files=True)
        ds_val = tfds.load('voc/2007', split='test')
    else:
        ds_train_a = tfds.load('voc/2007', split='train+validation', shuffle_files=True)
        ds_train_b = tfds.load('voc/2012', split='train+validation', shuffle_files=True)
        ds_train = ds_train_b.concat(ds_train_a)
        ds_val = tfds.load('voc/2007', split='test')

    return ds_train, ds_val


# if __name__ == '__main__':
#     pass
        





