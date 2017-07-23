from __future__ import print_function
from file_io import *
from get_model import VGG_16
import numpy as np
from os import listdir
from os.path import isfile, join


def get_weight_files(config):
    files_in_weight_dir = [f for f in listdir(config["weight_dir"]) if isfile(join(config["weight_dir"], f))]
    weight_files = [f.split(".hdf5")[0] for f in files_in_weight_dir if "weights-improvement" in f]
    return weight_files


def get_picture_names(config):
    # TODO: Read these from directory structure
    return [("person_and_bike_029", 1), ("no_person_001437", 0)]  # , ("person_100", 1), ("person_and_bike_190", 1)]


def unpack(vars_, activation_case, config):
    for key, value in config.items():
        if key not in vars_:
            vars_[key] = value

    vars_["user_defined_arr"] = [0]
    vars_["pkl_path"] = '/'.join([vars_["img_dir"], vars_["weight"], vars_["picture_name"], "pkls/"])

    if activation_case == 0:
        activation_fn = np.sum
        activation_metric = "_weight_sum_"
    elif activation_case == 1:
        activation_fn = np.var
        activation_metric = "_weight_var_"
    elif activation_case == 2:
        activation_fn = np.sum
        activation_metric = "_activation_sum_"
    elif activation_case == 3:
        activation_fn = np.var
        activation_metric = "_activation_var_"
    elif activation_case == 4:
        activation_fn = None
        activation_metric = "_correlation_"
    elif activation_case == 5:
        activation_fn = None
        activation_metric = "_precision_"
    else:
        activation_metric = ""
        activation_fn = None

    vars_["activation_metric"] = activation_metric
    vars_["activation_fn"] = activation_fn
    vars_["patch_path"] = '/'.join(
        ["writeup", vars_["weight"], vars_["picture_name"],
        activation_metric + str(vars_["user_defined_arr"][0]) + "/"])
    vars_["bounding_box_path"] = vars_["patch_path"] + "bounding_box"
    return vars_


def load_model(height, width, weights_path):
    """Loads the model, and specifies the optimizations

    Args:
        height: int -> height of the image coming in
        width: int -> width of the image coming in
        weights_path: string -> path to the weights to be used

    Returns:
        A keras model, having specified the optimizations

    """
    print("Loading model from {}...".format(weights_path), end='')
    model = VGG_16(height, width, weights_path)
    model.compile(optimizer="sgd", loss='categorical_crossentropy',
                  metrics=["accuracy"])
    return model


def format_array(arr):
    """
    Utility to format array for tiled plot

    args: arr (numpy array)
            shape : (n_samples, n_channels, img_dim1, img_dim2)
    """
    n_channels = arr.shape[1]
    len_arr = arr.shape[0]
    assert (n_channels == 1 or n_channels ==
            3), "n_channels should be 1 (Greyscale) or 3 (Color)"
    if n_channels == 1:
        arr = np.repeat(arr, 3, axis=1)

    shape1, shape2 = arr.shape[-2:]
    arr = np.transpose(arr, [1, 0, 2, 3])
    arr = arr.reshape([3, len_arr, shape1 * shape2]).astype(np.float64)
    arr = tuple([arr[i] for i in xrange(3)] + [None])
    return arr, shape1, shape2


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def top_two(x):
    """Returns the sum of the top two per list in x

    Args:
       x list of lists

    Returns:
        sum of the top 2
    """
    x = np.sort(x)
    x = x[::-1]
    return (x[0] + x[1])
