from __future__ import division

"""
ResNeXt is introduced in https://arxiv.org/abs/1611.05431
Implementation is inspired by https://github.com/prlz77/ResNeXt.pytorch
"""
from keras import backend as K

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation

from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def Conv2DGroup(filters, kernel_size, groups, strides=(1, 1), padding='valid',
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                kernel_regularizer=None, **kwargs):

    def get_slice(x, i, size):
        return x[:, :, :, size * i: size * (i + 1)]

    def f(inputs):
        split_filter = filters // groups
        split_input = K.int_shape(inputs)[-1] // groups
        output_convs = []

        if groups > 1:
            for i in range(groups):
                _slice = Lambda(get_slice,
                                arguments={'i': i,
                                        'size': split_input})(inputs)
                output_convs.append(Conv2D(filters=split_filter, kernel_size=kernel_size, 
                                    strides=strides, padding=padding, use_bias=use_bias, 
                                    kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    **kwargs)(_slice))
            outputs = concatenate(output_convs)
        else:
            outputs = Conv2D(filters=filters, kernel_size=kernel_size, 
                                strides=strides, padding=padding, use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                **kwargs)(inputs)
        return outputs

    return f

def bottleneck_c(filters, strides, cardinality, base_width, widen_factor):

    width_ratio = filters / (widen_factor * 64.)
    D = cardinality * int(base_width * width_ratio)

    def f(inputs):

        conv_reduce = Conv2D(filters=D, kernel_size=1, strides=1, padding='valid',
                             use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-4))(inputs)
        bn_reduce = BatchNormalization(axis=CHANNEL_AXIS)(conv_reduce)
        bn_reduce = Activation('relu')(bn_reduce)

        conv_conv = Conv2DGroup(filters=D, groups=cardinality, kernel_size=3, strides=strides, 
                                padding='same',
                                use_bias=False,
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(1e-4))(bn_reduce)
        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv_conv)
        norm = Activation('relu')(norm)

        conv_expand = Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid',
                             use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-4))(norm)
        bn_expand = BatchNormalization(axis=CHANNEL_AXIS)(conv_expand)


        if K.int_shape(inputs)[CHANNEL_AXIS] != filters:
            shortcut = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='valid',
                             use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-4))(inputs)
            residual = BatchNormalization(axis=CHANNEL_AXIS)(shortcut)
            return  Activation('relu')(add([residual, bn_expand]))

        else:
            return Activation('relu')(bn_expand)

    return f

class ResNeXt(object):

    def __init__(self, input_shape, num_class, cardinality, depth, base_width, widen_factor=4):

        self.input_shape = input_shape
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.num_class = num_class
        self.filters = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

    def build(self):

        _handle_dim_ordering()
                
        inputs = Input(shape=self.input_shape)
        conv_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                        use_bias=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(inputs)
        bn_1 = BatchNormalization(axis=CHANNEL_AXIS)(conv_1)
        init_block = Activation('relu')(bn_1)

        stage1 = self.block(self.stages[1], 1)(init_block)
        stage2 = self.block(self.stages[2], 2)(stage1)
        stage3 = self.block(self.stages[3], 2)(stage2)
        
        avg_pool = AveragePooling2D(pool_size=8, strides=1)(stage3)
        flatten = Flatten()(avg_pool)
        dense = Dense(units=self.num_class, kernel_initializer='he_normal',
                      activation='linear')(flatten)

        model = Model(inputs=inputs, outputs=dense)

        return model

    def block(self, filters, strides=2):

        def f(inputs):

            for bottleneck in range(self.block_depth):
                if bottleneck == 0:
                    block = bottleneck_c(filters, strides, self.cardinality, self.base_width,
                                            self.widen_factor)(inputs)
                else:
                    block = bottleneck_c(filters, 1, self.cardinality, self.base_width,
                                            self.widen_factor)(block)
            return block
        
        return f