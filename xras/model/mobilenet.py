import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, GlobalAveragePooling2D, Concatenate, DepthwiseConv2D
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.layers.experimental.preprocessing import Resizing
from keras.regularizers import l2
from keras.initializers import he_normal
import tensorflow as tf
import math

# MobileNet V1
# depth 固定
# MobileNet 使用较传统的后激活方式 (没有shortcut， 当然用后激活)
# 优化器 RMSprop
# weight decay: very little to depthwise conv
# 特性参数: width multiplier alpha, resolution multiplier rho

def MobileNet_V1(alpha=1.0, rho=1.0, weight_decay=0): # by default is 224*224

  weight_decay *= 0.5

  def cround(num):
    return math.floor(num + 0.5)
  
  def basic_block(x, out_filter, downsampling):

    x = DepthwiseConv2D((3, 3), (2, 2) if downsampling else (1, 1), padding='same',
               depthwise_initializer=he_normal,
              #  depthwise_regularizer=l2(weight_decay),
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x) 
    # x = Activation('relu')(x)
    x = tf.nn.relu6(x)

    x = Conv2D(out_filter, (1, 1), (1, 1), padding='same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x) 
    # x = Activation('relu')(x)
    x = tf.nn.relu6(x)

    return x
  
  def layer(x, n, block, out_filter, downsampling):
    x = block(x, out_filter, downsampling)
    for i in range(1, n):
      x = block(x, out_filter, False)
    return x

  num_filter = [64, 128, 256, 512, 1024]
  num_block = [1, 2, 2, 6, 2]
  num_filter = [cround(num * alpha) for num in num_filter]

  input = Input((32, 32, 3))

  x = Resizing(cround(224 * rho), cround(224 * rho))(input) # resolution multiplier

  x = Conv2D(32, (3, 3), (1, 1), 'same', # 注意：这里没有进行降采样，与原文中不同
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay),
             use_bias=False)(x) # ->batch_size, (224, 224), 16
  x = BatchNormalization(momentum=0.9)(x) 
  # x = Activation('relu')(x)
  x = tf.nn.relu6(x)
  
  x = layer(x, num_block[0], basic_block, num_filter[0], False) # ->batch_size, (112, 112), 64
  x = layer(x, num_block[1], basic_block, num_filter[1], True) # ->batch_size, (56, 56), 128
  x = layer(x, num_block[2], basic_block, num_filter[2], True) # ->batch_size, (28, 28), 256
  x = layer(x, num_block[3], basic_block, num_filter[3], True) # ->batch_size, (14, 14), 512
  x = layer(x, num_block[4], basic_block, num_filter[4], True) # ->batch_size, (7, 7), 1024

  x = GlobalAveragePooling2D()(x) # ->batch_size, 1024
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay),
            use_bias=False)(x)

  model = Model(input, x)

  return model