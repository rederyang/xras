import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, GlobalAveragePooling2D, Concatenate
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.initializers import he_normal
import math

# DenseNet
# l: layers, k: growth rate, theta: compression factor
# for DenseNet BC type: bottleneck=True, 0 < theta < 1
# DenseNet的block较特殊，与一般的resnet相比少一层conv

def DenseNet(l=40, k=12, bottleneck=False, theta=1, weight_decay=1e-4):

  n = (l - 4) // 3
  if bottleneck:
    n //= 2
  bc = bottleneck and theta < 1 and 0 < theta
  weight_decay *= 0.5

  def dense_block(x, k): # ResNet V2 style (preact)
    for _ in range(n):
      o = x
      if bottleneck:
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Conv2D(4*k, (1, 1), (1, 1),
                   kernel_initializer=he_normal,
                   kernel_regularizer=l2(weight_decay),
                   use_bias=False)(x)
      x = BatchNormalization(momentum=0.9)(x)
      x = Activation('relu')(x)
      x = Conv2D(k, (3, 3), (1, 1), 'same',
                 kernel_initializer=he_normal,
                 kernel_regularizer=l2(weight_decay),
                 use_bias=False)(x)
      x = Concatenate(-1)([o, x]) 
    return x

  def transition_layer(x, num_filter):
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filter, (1, 1), (1, 1),
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    x = AveragePooling2D((2, 2))(x)
    return x

  input = Input((32, 32, 3))

  num_filter = 0
  x = Conv2D(2*k if bc else 16, (3, 3), (1, 1), 'same', 
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay),
             use_bias=False)(input) # ->batch_size, (32, 32), 16 or 2k
  num_filter += 2*k if bc else 16

  x = dense_block(x, k) # ->batch_size, (32, 32), ~+n*k
  num_filter = math.floor((num_filter + n*k) * theta)

  x = transition_layer(x, num_filter)

  x = dense_block(x, k) # ->batch_size, (32, 32), ~+2*n*k
  num_filter = math.floor((num_filter + n*k) * theta)

  x = transition_layer(x, num_filter)

  x = dense_block(x, k) # ->batch_size, (32, 32), ~+3*n*k
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)

  x = GlobalAveragePooling2D()(x) # ->batch_size, ~+3*n*k
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay),
            use_bias=False)(x)

  model = Model(input, x)

  return model