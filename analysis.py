from __future__ import print_function
#from file_io import get_predicted_prob_filename
import numpy as np
np.set_printoptions(threshold=np.inf)
import get_model
from keras.utils import np_utils


def calc_accuracy(X_train, X_test, Y_train, Y_test):
    """
    Args:
        X_train:
        X_test:
        Y_train:
        Y_test:
    Returns:
        float value representing the accuracy
    """
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    model = get_model.get_simple_cnn(
        height=X_train.shape[2], width=X_train.shape[3])
    print("Fitting Simple CNN model to X_train of shape {} and Y_train of shape {}".format(X_train.shape, Y_train.shape))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(X_train, Y_train,
              batch_size=8, nb_epoch=5, verbose=1)
    print("Scoring Simple CNN model on X_test of shape {} and Y_test of shape {}.".format(X_test.shape, Y_test.shape))
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("Score: {}".format(score))
    return score


def calc_overlap(l1, l2):
    """l1 and l2 are lists of [layer_name, lists of neurons]
    Args:
        l1:
        l2:

    Returns:
        scalar value representing the jaccard similarity between the two sets
    """
    assert(len(l1) == len(l2))
    same = 0
    total = 0
    for [l1_name, layer1], [l2_name, layer2] in zip(l1, l2):
        print("Layer1: {}".format(layer1))
        print("Layer2: {}".format(layer2))
        total += len(layer1) + len(layer2)
        for l in layer1:
            if l in layer2:
                same += 1
    return same / float(total)
