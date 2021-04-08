# Inception

'''TODO: 将aux_classifier改进为原文中的结构'''

from keras.models import Model, Input
from keras.layers import Conv2D, Dropout, Dense, Activation, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.initializers import he_normal

def Inception_V2(weight_decay=1e-4):
  weight_decay *= 0.5  

  def inception(x, num_filters, pooling='avg', downsampling=False): # params: a list of (len=6 or 5) num_filters within the inception module

    if num_filters[0]:
      x1 = Conv2D(num_filters[0], (1, 1), (1, 1) if not downsampling else (2, 2), padding='same',
                  kernel_initializer=he_normal,
                  kernel_regularizer=l2(weight_decay))(x) 
      x1 = BatchNormalization(momentum=0.9)(x1)
      x1 = Activation('relu')(x1)
    
    x2 = Conv2D(num_filters[1], (1, 1), (1, 1), padding='same')(x)
    x2 = BatchNormalization(momentum=0.9)(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(num_filters[2], (3, 3), (1, 1) if not downsampling else (2, 2), padding='same',
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay))(x2) 
    x2 = BatchNormalization(momentum=0.9)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(num_filters[3], (1, 1), (1, 1), padding='same',
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay))(x)
    x3 = BatchNormalization(momentum=0.9)(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(num_filters[4], (3, 3), (1, 1), padding='same',
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay))(x3)
    x3 = BatchNormalization(momentum=0.9)(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(num_filters[4], (3, 3), (1, 1) if not downsampling else (2, 2), padding='same',
                kernel_initializer=he_normal,
                kernel_regularizer=l2(weight_decay))(x3) 
    x3 = BatchNormalization(momentum=0.9)(x3)
    x3 = Activation('relu')(x3)

    if pooling == 'avg':
      x4 = AveragePooling2D((3, 3), (1, 1), padding='same')(x)
      x4 = Conv2D(num_filters[5], (1, 1), (1, 1), padding='same', 
                  kernel_initializer=he_normal,
                  kernel_regularizer=l2(weight_decay))(x4)
      x4 = BatchNormalization(momentum=0.9)(x4)
      x4 = Activation('relu')(x4)
    elif downsampling == True:
      x4 = MaxPooling2D((3, 3), (2, 2), padding='same')(x)
    else:
      x4 = MaxPooling2D((3, 3), (1, 1), padding='same')(x)
      x4 = Conv2D(num_filters[5], (1, 1), (1, 1), padding='same', 
                  kernel_initializer=he_normal,
                  kernel_regularizer=l2(weight_decay))(x4)
      x4 = BatchNormalization(momentum=0.9)(x4)
      x4 = Activation('relu')(x4)

    return Concatenate(-1)([x1 ,x2, x3, x4]) if num_filters[0] else Concatenate(-1)([x2, x3, x4])

  # for cifar10, the aux classifiers are modified to 1*1*128(->4*4*128), GlobalAverage, Dense(1024), Relu, Dense(100, 'softmax'), 
  def aux_classifier(x, name): 
    x = Conv2D(128, (1, 1),
               kernel_initializer=he_normal,
               kernel_regularizer=l2(weight_decay))(x, training=True)
    x = BatchNormalization(momentum=0.9)(x, training=True)
    x = Activation('relu')(x, training=True)
    x = GlobalAveragePooling2D()(x, training=True)
    x = Dense(1024)(x, training=True)
    x = BatchNormalization(momentum=0.9)(x, training=True)
    x = Activation('relu')(x, training=True)
    x = Dense(10, activation='softmax', name=name)(x, training=True)
    return x

  input = Input((224, 224, 3), name='input_image') # 32, 32, 3

  x = Conv2D(64, (7, 7), (2, 2), padding='same',
             kernel_initializer=he_normal, # 32, 32, 64
             kernel_regularizer=l2(weight_decay))(input)
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), (2, 2), padding='same')(x) # 16, 16, 64
  x = Conv2D(64, (1, 1), (1, 1), padding='same', # 16, 16, 64
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay))(x)
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)
  x = Conv2D(192, (3, 3), (1, 1), padding='same', # 16, 16, 192
             kernel_initializer=he_normal,
             kernel_regularizer=l2(weight_decay))(x)
  x = BatchNormalization(momentum=0.9)(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), (2, 2), padding='same')(x) # 8, 8, 192

  # inception 3
  x = inception(x, [64] * 4 + [96, 32])
  x = inception(x, [64] * 2 + [96, 64] * 2)
  x = inception(x, [0, 128, 160, 64, 96], 'max', True)

  # inception 4
  x = inception(x, [224, 64] + [96]*2 + [128] * 2)
  aux_output1 = aux_classifier(x, 'aux_output1')
  x = inception(x, [192, 96, 128, 96] + [128] * 2)
  x = inception(x, [160, 128, 128, 128, 160, 128]) # note that in the paper here is [160, 128] * 3
  x = inception(x, [96, 128, 160, 160, 192, 128]) # note that in the paper here is [96, 128, 192, 160, 192, 128]
  aux_output2 = aux_classifier(x, 'aux_output2')
  x = inception(x, [0, 128, 192, 192, 256], 'max', True)

  # inception 5
  x = inception(x, [352, 192, 320, 160, 224, 128])
  x = inception(x, [352, 192, 320, 160, 224, 128], 'max', False)

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, activation='softmax', name='main_output')(x)

  model = Model(input, [x, aux_output1, aux_output2])

  return model  