import os
import random
import numpy as np
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def view_model(model, file_name=None):
    plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=False)

def plot_log(log, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xticks=None, yticks=None, # log: dict like {'legend': log}
            savefig=None, figsize=(4,3), dpi=160):
    plt.figure(figsize=(4,3), dpi=160)
    for seq in log.values():
        plt.plot(seq)
    plt.legend(log.keys(), loc='upper left')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid()
    plt.savefig(savefig)
    plt.show()