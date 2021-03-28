import os
import random
import numpy as np
import numpy as np
import tensorflow as tf
from keras.utils import plot_model

import matplotlib.pyplot as plt

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def v_model(model, file_name=None):
    plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=False)


# def plot_result(log, yticks=np.arange(0., 1., 0.1), acc_to_error=False): # log: a tuple contains train log (and val log if exists)
#     if acc_to_error:
#         for i in range(len(log)):
#             log[i] = [1.0-x for x in log[i]]

#     plt.figure(figsize=(10, 7.5))
#     for i in range(len(log)):
#         plt.plot(log)  
#     plt.title('acc')
#     plt.xlabel('epoch')
#     plt.ylabel('acc')
#     plt.yticks(yticks)
#     plt.
#     plt.show()