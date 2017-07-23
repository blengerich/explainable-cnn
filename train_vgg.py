
# coding: utf-8

# In[*]

import KerasDeconv
from utils import plot_deconv
import os
from get_model import VGG_16
#import cPickle as pkl
from components.component_predictor import run_predictor, reconstruct_data
import numpy as np
import sys
#import cPickle as pkl
from keras import backend as K
from PIL import Image, ImageDraw
from scipy import stats
import os
from run_deconv import load_and_init

height=128
width=128


import time
start = time.time()

train_neg_x = load_and_init("INRIAPerson/Train/neg", height, width)
train_neg_y = np.hstack((np.ones((train_neg_x.shape[0], 1)), np.zeros((train_neg_x.shape[0], 1))))
#train_neg_y = np.zeros((train_neg_x.shape[0], 1))
print(train_neg_x.shape)
print(train_neg_y.shape)

train_pos_x = load_and_init("INRIAPerson/Train/pos", height, width)
train_pos_y = np.hstack((np.zeros((train_pos_x.shape[0], 1)), np.ones((train_pos_x.shape[0], 1))))
#train_pos_y = np.ones((train_pos_y.shape[0], 1)) 
print(train_pos_x.shape)
print(train_pos_y.shape)
print("Loaded all images, took {:2f} seconds".format(time.time() - start))


# In[*]

X_train = np.vstack((train_neg_x, train_pos_x))
Y_train = np.vstack((train_neg_y, train_pos_y))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
Y_train = Y_train.reshape((-1, 2))
print(X_train.shape)
print(Y_train.shape)


# In[*]

#from run_deconv import load_model
#vgg = load_model(height, width, './Data/cam_checkpoint.hdf5')
#vgg = load_model(height, width, None)

vgg = VGG_16(height, width, None)
vgg.compile(optimizer="sgd", loss='categorical_crossentropy',
              metrics=["accuracy"])
#model.compile(optimizer="sgd", loss='sparse_categorical_crossentropy',
#              metrics=["accuracy"])


# In[*]

from keras.callbacks import ModelCheckpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
hist = vgg.fit(X_train, Y_train, nb_epoch=1, verbose=1, callbacks=callbacks_list)


# In[*]



