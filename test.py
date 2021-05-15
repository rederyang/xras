import matplotlib.pyplot as plt
from xras.data.dataset import load_voc_dataset

ds_train, ds_test = load_voc_dataset()

for example in ds_train.take(1):
    print(example)





