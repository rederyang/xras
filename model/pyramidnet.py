import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, GlobalAveragePooling2D, Concatenate
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.initializers import he_normal
import math

# PyramidNet

# TODO：参数量与论文中结果有微小的差异

def PyramidNet(depth=110, alpha=48, bottleneck=False, weight_decay=1e-4):

  N = (depth - 2) // 3 if bottleneck else (depth - 2) // 2 # N = N_2 + N_3 + N_4
  n = N // 3 # num of residual units of each stage
  step = alpha / N 
  weight_decay *= 0.5

  def round(num):
    return math.floor(num+0.5)

  def padded_shortcut(x, in_filter, out_filter, mapsize, downsampling):
    if downsampling:
      x = AveragePooling2D((2, 2), (2, 2))(x) # 参考官方实现 
    to_pad = K.zeros_like(x[:, :, :, 0:1])
    for _ in range(out_filter - in_filter): # 在channel维度上填充全零的channel以匹配
      x = Concatenate(-1)([x, to_pad])
    
    # padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, out_filter - in_filter]]) # 在channel维度上concatenate全零的channel
    # return tf.pad(x, padding, 'CONSTANT', 0.)
    return x
  
  def basic_block(x, mapsize, downsampling):
    nonlocal cur_channels
    tmp_channels = round(channels)

    o = x

    x = BatchNormalization(momentum=0.9)(x)
    x = Conv2D(tmp_channels, (3, 3), (2, 2) if downsampling else (1, 1), padding='same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(tmp_channels, (3, 3), (1, 1), padding='same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)

    x = Add()([x, padded_shortcut(o, cur_channels, tmp_channels, mapsize, downsampling)])

    cur_channels = tmp_channels

    return x


  def bottleneck_block(x, mapsize, downsampling):
    nonlocal cur_channels
    tmp_channels = round(channels)

    o = x

    x = BatchNormalization(momentum=0.9)(x)
    x = Conv2D(tmp_channels, (1, 1), (1, 1), padding='same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(tmp_channels, (3, 3), (2, 2) if downsampling else (1, 1), padding='same', # 在中间的conv进行下采样(参考官方实现）
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2D(tmp_channels * 4, (1, 1), (1, 1), padding='same',
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    
    x = Add()([x, padded_shortcut(o, cur_channels, tmp_channels * 4, mapsize, downsampling)])
    
    cur_channels = tmp_channels * 4

    return x
  
  def layer(x, block, mapsize, downsampling):
    nonlocal channels
    channels += step
    x = block(x, mapsize, downsampling)
    for i in range(1, n):
      channels += step
      x = block(x, mapsize, False)
    return x

  mapsize = [32, 16, 8]

  input = Input((32, 32, 3))

  cur_channels = 0 # 用于描述block之间的channel数量
  channels = 16
  x = Conv2D(channels, (3, 3), (1, 1), 'same',
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay),
             use_bias=False)(input) # ->batch_size, (32, 32), 16
  x = BatchNormalization(momentum=0.9)(x) # 参考官方实现
  cur_channels = channels
  
  x = layer(x, bottleneck_block if bottleneck else basic_block, mapsize[0], False) # ->batch_size, (32, 32), 16 + alpha * (1/3)
  x = layer(x, bottleneck_block if bottleneck else basic_block, mapsize[1], True) # ->batch_size, (16, 16), 16 + alpha * (2/3)
  x = layer(x, bottleneck_block if bottleneck else basic_block, mapsize[2], True) # ->batch_size, (8, 8), 16 + alpha
  x = BatchNormalization(momentum=0.9)(x) # 参考官方实现
  x = Activation('relu')(x) # 参考官方实现

  assert(round(channels) == 16 + alpha) # 在以上构造结束后，应有该等式

  x = GlobalAveragePooling2D()(x) # ->batch_size, num_filter[2]
  x = Dense(10, activation='softmax',
            kernel_initializer=he_normal,
            kernel_regularizer=l2(weight_decay),
            use_bias=False)(x)

  model = Model(input, x)

  return model