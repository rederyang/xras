import matplotlib.pyplot as plt
from xras.data.datasets import load_voc_dataset

ds_train, ds_test = load_voc_dataset()

for example in ds_train.take(5):
    print(example)





