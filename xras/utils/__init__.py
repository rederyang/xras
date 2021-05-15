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
    if file_name:
        plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=False)
    else:
        plot_model(model, show_shapes=True)

def plot_log(log, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xticks=None, yticks=None, # log: dict like {'legend': log}
            savefig=None, figsize=(4,3), dpi=160):
    plt.figure(figsize=(4,3), dpi=160)
    for seq in log.values():
        plt.plot(seq)
    plt.legend(log.keys(), loc='upper left')
    plt.title(title)
    plt.ylabel(ylabel if ylabel else title)
    plt.xlabel(xlabel if xlabel else 'Epoch')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid()
    plt.savefig(savefig)
    plt.show()

def plot_from_history(history, acc_to_error=False):

    if acc_to_error:
        error={
        'train':[1 - x for x in history.history['accuracy']],
        'val':[1 - x for x in history.history['val_accuracy']] 
        }
        plot_log(error, title='Error', yticks=np.arange(0., 1., 0.1), savefig='error.jpg')
    else:
        acc={
        'train':history.history['accuracy'],
        'val':history.history['val_accuracy']
        }
        plot_log(acc, title='Accuracy', yticks=np.arange(0., 1., 0.1), savefig='acc.jpg')

    loss={
        'train':history.history['loss'],
        'val':history.history['val_loss']    
    }
    
    plot_log(loss, title='Loss', savefig='loss.jpg')