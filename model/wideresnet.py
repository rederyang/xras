# Wide ResNet

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, SpatialDropout2D, Add, Input, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2
from keras.initializers import he_normal

def WideResNet(depth=28, k=1, weight_decay=5e-4, dropout=0.3):

  n = (depth - 4) // (2 * 3) # depth - 4 （conv1 + dense + 2 * proj）
  weight_decay *= 0.5

  def shortcut(x, input_channels, output_channels, downsampling):
    if input_channels != output_channels:
      x = Conv2D(output_channels, (1, 1), (2, 2) if downsampling else (1, 1),
                 kernel_initializer=he_normal,
                 kernel_regularizer=l2(weight_decay))(x)
    return x

  def basic(input, input_channels, output_channels, downsampling):

    x = BatchNormalization(momentum=0.9)(input)
    x = Activation('relu')(x)

    if input_channels != output_channels: # 注意：对于proj shortcut，需要对两条支路的输入都进行act
      input = x

    x = Conv2D(output_channels, (3, 3), (2, 2) if downsampling else (1, 1), padding='same', 
              kernel_initializer=he_normal, 
              kernel_regularizer=l2(weight_decay),
              use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = SpatialDropout2D(dropout)(x) # 注意：这里是先act再dropout

    x = Conv2D(output_channels, (3, 3), (1, 1), padding='same', 
              kernel_initializer=he_normal, 
              kernel_regularizer=l2(weight_decay),
              use_bias=False)(x)

    merge = Add()([x, shortcut(input, input_channels, output_channels, downsampling)])

    return merge

  def stage(x, input_channels, output_channels, downsampling):
    x = basic(x, input_channels, output_channels, downsampling)
    for i in range(1, n):
      x = basic(x, output_channels, output_channels, False)
    return x

  input = Input((32, 32, 3))

  output_channels = [16*k, 32*k, 64*k] # output channels of each stage

  x = Conv2D(16, (3, 3), (1, 1), padding='same',
            kernel_initializer=he_normal, 
            kernel_regularizer=l2(weight_decay),
            use_bias=False)(input)
  x = stage(x, 16, output_channels[0], False)
  x = stage(x, output_channels[0], output_channels[1], True)
  x = stage(x, output_channels[1], output_channels[2], True)
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_regularizer=l2(weight_decay),
            kernel_initializer=he_normal,
            activation='softmax')(x)

  model = Model(input, x)

  return model