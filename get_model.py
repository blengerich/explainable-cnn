# -*- coding: utf-8 -*-
'''VGG16 model for Keras.
# Reference:
'''
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def Alexnet(height, width, weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, height, width)))
    model.add(Convolution2D(64, 11, 11, border_mode="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, 7, 7, border_mode="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(192, 3, 3, border_mode="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(256, 3, 3, border_mode="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, init='normal', activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(512, init='normal', activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(2, init='normal', activation="softmax"))

    if weights_path:
        print("Loading weights...", end='\t')
        model.load_weights(weights_path)
        print("Finished.")

    return model


def VGG_16(height, width, weights_path=None):
    """
    VGG Model Keras specification
    args: weights_path (str) trained weights file path
    returns model (Keras model)
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, height, width)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(2, activation="softmax", init="uniform"))

    if weights_path:
        print("Loading weights...", end='\t')
        model.load_weights(weights_path)
        print("Finished.")

    return model


def get_simple_cnn(height, width):
    """ A simple CNN that has the same input/output shapes as the VGG16 model.

    Args:
        height: input height
        width: input width
    Return: Keras model

    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, height, width)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((4, 4), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((4, 4), strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(MaxPooling2D((4, 4), strides=(2, 2)))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(2, activation="softmax", init="uniform"))
    return model
