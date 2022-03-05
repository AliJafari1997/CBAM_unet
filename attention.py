import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class channel_attention(tf.keras.layers.Layer):
    """ 
    channel attention module 
    """
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(channel_attention, self).get_config().copy()
        config.update({
            'ratio': self.ratio
        })
        return config

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channel // self.ratio,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)    
        avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])

class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)


    def get_config(self):
        config = super(spatial_attention, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)	
            
        return tf.keras.layers.multiply([inputs, feature])


def ch_attention(x, ratio = 8):
    x = channel_attention(ratio = ratio)(x)
    return x


def s_attention(x, kernel_size=7):
    x = spatial_attention(kernel_size=kernel_size)(x)
    return x


def CBAM(x, ratio=8, kernel_size=7):
    ch_attention = channel_attention(ratio = ratio)
    s_attention = spatial_attention(kernel_size=kernel_size)
    x = ch_attention(x)
    x = s_attention(x)
    return x


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x



def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
            config = super(PAM, self).get_config().copy()
            config.update({
                'ratio': self.ratio,
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            })
            return config



    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input
        return out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint


    def get_config(self):
        config = super(CAM, self).get_config().copy()
        config.update({
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        })
        return config
    
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma*aaTa + input
        return out


def dual_attention(x):
    pam = PAM()(x)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    pam = BatchNormalization(axis=3)(pam)
    pam = Activation('relu')(pam)
    pam = Dropout(0.5)(pam)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

    cam = CAM()(x)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    cam = BatchNormalization(axis=3)(cam)
    cam = Activation('relu')(cam)
    cam = Dropout(0.5)(cam)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)

    feature_sum = add([pam, cam])
    feature_sum = Dropout(0.5)(feature_sum)

    x = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x
