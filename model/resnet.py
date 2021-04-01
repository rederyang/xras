# resnet for cifar10

import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, GlobalAveragePooling2D, Concatenate
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.initializers import he_normal

# resnet V1 for cifar10
def Resnet_V1(depth=20, weight_decay=1e-4):
  
  n = (depth - 2) // 6
  weight_decay = 0.5 * weight_decay # featuring keras 

  def shortcut(x, downsampling):
    if downsampling:
      x = MaxPooling2D((1, 1), (2, 2), 'valid')(x) # 对原输入进行下采样，即忽略一半的值
      x = Concatenate(-1)([x, K.zeros_like(x)]) # 在channel维度上concatenate全零的channel
    return x

  def basic_block(x, num_filter, downsampling):
    input = x
    if downsampling: # 下采样的同时倍增通道数
      x = Conv2D(num_filter, (3, 3), (2, 2), 'same',
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay),
                 bias_regularizer=l2(weight_decay))(x)
    else:
      x = Conv2D(num_filter, (3, 3), (1, 1), 'same',
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay),
                 bias_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filter, (3, 3), (1, 1), 'same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Add()([x, shortcut(input, downsampling)])
    x = Activation('relu')(x)
    return x

  def layer(x, block, num_filter):
    for i in range(n):
      x = block(x, num_filter, i==0 and num_filter!=16) # 若不为第一个layer且为第一个block，则进行下采样
    return x

  # output_map_size = [32, 16, 8]
  num_filter = [16, 32, 64]

  input = Input((32, 32, 3))

  x = Conv2D(num_filter[0], (3, 3), (1, 1), 'same',
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay),
             bias_regularizer=l2(weight_decay))(input) # ->batch_size, (32, 32), num_filter[0]
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)

  x = layer(x, basic_block, num_filter[0]) # ->batch_size, (32, 32), num_filter[0]
  x = layer(x, basic_block, num_filter[1]) # ->batch_size, (16, 16), num_filter[1]
  x = layer(x, basic_block, num_filter[2]) # ->batch_size, (8, 8), num_filter[2]
  x = GlobalAveragePooling2D()(x) # ->batch_size, num_channel
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay),
            bias_regularizer=l2(weight_decay))(x)

  model = Model(input, x)

  return model

# resnetV2 without bottleneck structure
def Resnet_V2(depth=20, weight_decay=1e-4):

  n = (depth - 2) // 6
  weight_decay *= 0.5

  def shortcut_A(x, in_filter, out_filter):
    if in_filter != out_filter:
      x = MaxPooling2D((1, 1), (2, 2))(x) # when mismatch, downsample and padded with zero channel
      x = Concatenate(-1)([x, K.zeros_like(x)]) 
    return x

  def basic_block(x, in_filter, out_filter, downsampling, type=None):
    o = x
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    if type == 'first':
      o = x 
    x = Conv2D(out_filter, (3, 3), (2, 2) if downsampling else (1, 1), 'same',
              kernel_initializer=he_normal,
              kernel_regularizer=l2(weight_decay),
                bias_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(out_filter, (3, 3), (1, 1), 'same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay))(x)
    x = Add()([x, shortcut_A(o, in_filter, out_filter)])
    return x

  def layer(x, block, in_filter, out_filter, downsampling, type=None):
    x = block(x, in_filter, out_filter, downsampling, type)
    for i in range(1, n):
      x = block(x, out_filter, out_filter, False)
    return x

  num_filter = [16, 32, 64]
  input = Input((32, 32, 3))

  x = Conv2D(16, (3, 3), (1, 1), 'same',
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay))(input) # ->batch_size, (32, 32), num_filter[0]

  x = layer(x, basic_block, 16, num_filter[0], False, 'first') # ->batch_size, (32, 32), 16
  x = layer(x, basic_block, num_filter[0], num_filter[1], True) # ->batch_size, (16, 16), 32
  x = layer(x, basic_block, num_filter[1], num_filter[2], True) # ->batch_size, (8, 8), 64
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)

  x = GlobalAveragePooling2D()(x) # ->batch_size, num_filter[2]
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay))(x)

  model = Model(input, x)

  return model

# resnetV2 with bottleneck structure

def Resnet_V2_bottleneck(depth=29, weight_decay=1e-4):

  n = (depth - 2) // 9
  weight_decay *= 0.5

  def shortcut_B(x, in_filter, out_filter, downsampling):
    if in_filter != out_filter: # proj to match channel dimension
      x = Conv2D(out_filter, (1, 1), (2, 2) if downsampling else (1, 1),
                 kernel_initializer=he_normal,
                 kernel_regularizer=l2(weight_decay))(x)
    return x

  def bottleneck(x, in_filter, out_filter, downsampling):
    o = x
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x) # 进行了proj，所以在split之前act
    if in_filter != out_filter:
      o = x
    
    x = Conv2D(out_filter / 4, (1, 1), (2, 2) if downsampling else (1, 1),
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(out_filter / 4, (3, 3), (1, 1), padding='same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(out_filter, (1, 1), (1, 1),
              kernel_initializer=he_normal,
              kernel_regularizer=l2(weight_decay))(x)
    
    x = Add()([x, shortcut_B(o, in_filter, out_filter, downsampling)])
    return x
  
  def layer(x, block, in_filter, out_filter, downsampling):
    x = block(x, in_filter, out_filter, downsampling)
    for i in range(1, n):
      x = block(x, out_filter, out_filter, False)
    return x

  num_filter = [64, 128, 256]

  input = Input((32, 32, 3))

  x = Conv2D(16, (3, 3), (1, 1), 'same',
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay))(input) # ->batch_size, (32, 32), num_filter[0]

  x = layer(x, bottleneck, 16, num_filter[0], False) # ->batch_size, (32, 32), 64
  x = layer(x, bottleneck, num_filter[0], num_filter[1], True) # ->batch_size, (16, 16), 128
  x = layer(x, bottleneck, num_filter[1], num_filter[2], True) # ->batch_size, (8, 8), 256
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)

  x = GlobalAveragePooling2D()(x) # ->batch_size, num_filter[2]
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay))(x)

  model = Model(input, x)

  return model


def ResNeXt(depth=29, c=8, weight_decay=5e-4):

  n = (depth - 2) // 9
  weight_decay *= 0.5

  def shortcut_B(x, in_filter, out_filter, downsampling):
    if in_filter != out_filter: # proj to match channel dimension
      x = Conv2D(out_filter, (1, 1), (2, 2) if downsampling else (1, 1),
                 kernel_initializer=he_normal,
                 kernel_regularizer=l2(weight_decay),
                 use_bias=False)(x)
    return x

  def group_bottleneck(x, in_filter, out_filter, downsampling):
    o = x
    
    # conv 1x1
    x = Conv2D(out_filter * c / 4, (1, 1), (1, 1),
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay),
                use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    # conv 3x3 与resnetV1不同：group conv
    x = Conv2D(out_filter * c / 4, (3, 3), (2, 2) if downsampling else (1, 1), 'same', # 与resnet V1不同：在conv3x3进行降采样
               groups = c,
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    # conv 1x1
    x = Conv2D(out_filter, (1, 1), (1, 1),
              kernel_initializer=he_normal,
              kernel_regularizer=l2(weight_decay),
              use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Add()([x, shortcut_B(o, in_filter, out_filter, downsampling)])
    x = Activation('relu')(x)

    return x
  
  def layer(x, block, in_filter, out_filter, downsampling):
    x = block(x, in_filter, out_filter, downsampling)
    for i in range(1, n):
      x = block(x, out_filter, out_filter, False)
    return x

  num_filter = [256, 512, 1024] # 1st block template: 1x1 64; 3x3 64; 1x1 256

  input = Input((32, 32, 3))

  x = Conv2D(64, (3, 3), (1, 1), 'same', # 与resnet V1不同，第一步conv即为64个channel
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay),
             use_bias=False)(input) # ->batch_size, (32, 32), num_filter[0]
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)

  x = layer(x, group_bottleneck, 64, num_filter[0], False) # ->batch_size, (32, 32), num_filter[0]
  x = layer(x, group_bottleneck, num_filter[0], num_filter[1], True) # ->batch_size, (16, 16), num_filter[1]
  x = layer(x, group_bottleneck, num_filter[1], num_filter[2], True) # ->batch_size, (8, 8), num_filter[2]

  x = GlobalAveragePooling2D()(x) # ->batch_size, num_filter[2]
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay))(x)

  model = Model(input, x)

  return model

