# # resnet_v1 for cifar10
# # Option A shortcut: padded with zero valued channels when increasing the num of channels

# import keras.backend as K
# from keras.models import Model
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, GlobalAveragePooling2D, Concatenate
# from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
# from keras.regularizers import l2
# from keras.initializers import he_normal

# def Resnet_V1(depth=20, weight_decay=1e-4):
  
#   n = (depth - 2) // 6
#   weight_decay = 0.5 * weight_decay # featuring keras 

#   def shortcut(x, downsampling):
#     if downsampling:
#       x = MaxPooling2D((1, 1), (2, 2), 'valid')(x) # 对原输入进行下采样，即忽略一半的值
#       x = Concatenate(-1)([x, K.zeros_like(x)]) # 在channel维度上concatenate全零的channel
#     return x

#   def basic_block(x, num_filter, downsampling):
#     input = x
#     if downsampling: # 下采样的同时倍增通道数
#       x = Conv2D(num_filter, (3, 3), (2, 2), 'same',
#                 kernel_initializer=he_normal,
#                 kernel_regularizer=l2(weight_decay),
#                  bias_regularizer=l2(weight_decay))(x)
#     else:
#       x = Conv2D(num_filter, (3, 3), (1, 1), 'same',
#                 kernel_initializer=he_normal,
#                 kernel_regularizer=l2(weight_decay),
#                  bias_regularizer=l2(weight_decay))(x)
#     x = BatchNormalization(momentum=0.9)(x)
#     x = Activation('relu')(x)
#     x = Conv2D(num_filter, (3, 3), (1, 1), 'same',
#                kernel_initializer=he_normal,
#                kernel_regularizer=l2(weight_decay),
#                bias_regularizer=l2(weight_decay))(x)
#     x = BatchNormalization(momentum=0.9)(x)
#     x = Add()([x, shortcut(input, downsampling)])
#     x = Activation('relu')(x)
#     return x

#   def layer(x, block, num_filter):
#     for i in range(n):
#       x = block(x, num_filter, i==0 and num_filter!=16) # 若不为第一个layer且为第一个block，则进行下采样
#     return x

#   # output_map_size = [32, 16, 8]
#   num_filter = [16, 32, 64]

#   input = Input((32, 32, 3))

#   x = Conv2D(num_filter[0], (3, 3), (1, 1), 'same',
#              kernel_initializer=he_normal,
#              kernel_regularizer=l2(weight_decay),
#              bias_regularizer=l2(weight_decay))(input) # ->batch_size, (32, 32), num_filter[0]

#   x = layer(x, basic_block, num_filter[0]) # ->batch_size, (32, 32), num_filter[0]
#   x = layer(x, basic_block, num_filter[1]) # ->batch_size, (16, 16), num_filter[1]
#   x = layer(x, basic_block, num_filter[2]) # ->batch_size, (8, 8), num_filter[2]
#   x = GlobalAveragePooling2D()(x) # ->batch_size, num_channel
#   x = Dense(10, activation='softmax',
#             kernel_initializer=he_normal,
#             kernel_regularizer=l2(weight_decay),
#             bias_regularizer=l2(weight_decay))(x)

#   model = Model(input, x)

#   return model