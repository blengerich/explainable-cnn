from __future__ import print_function
import cPickle as pkl
from image_utils import tile_raster_images, rgb_to_gbr, crop_images
from file_io import make_dir#, get_pkl_filename
import numpy as np
from utils import format_array
import matplotlib.pyplot as plt


def load_or_create_patches(image_data, target_layers, Dec,
                           save_pkl=True, pkl_dir="",
                           save_raw=False, raw_dir=""):
    """ Tries to load the patches for the model with given weight filename.
    If this fails, it deconvolves the network and finds the coordinates of the
    patches corresponding to each neuron.

    Args:
        image_data:
        target_layers:
        Dec:
        save_pkl:
        pkl_dir:
        pkl_filename:
        save_raw:
        raw_dir:

    Returns:
        patches: {(layer_id, neuron_id): [min_x, max_x, min_y, max_y]]}

    """
    pkl_filename = "{}/patches.pkl".format(pkl_dir)
    patches = load_patches(pkl_filename)  # TODO: delete patches, we are storing these differently now

    if patches is None:
        patches = create_patches(
            image_data, target_layers, Dec, save_raw, raw_dir, show=False)

        if save_pkl and pkl_dir is not "":
            print("Saving patches pkl to {}".format(pkl_filename))
            make_dir(pkl_dir)
            with open(pkl_filename, "wb") as f:
                pkl.dump(patches, f)

    return patches


def load_patches(filename):
    """
    Args:
        filename: where the pkl file should be located

    Returns:
        unpickled patches

    """
    try:
        with open(filename, "rb") as f:
            return pkl.load(f)
    except IOError:
        return None


def create_patches(image_data, target_layers, Dec,
                   save_raw=False, raw_dir="", show=False):
    """ Deconvoles the network and finds the coordinates of the patches
    corresponding to each neuron in the network.

    Args:
        image_data:
        target_layers:
        Dec:
        save_raw:
        raw_dir:
        show:
        save_pkl:
        pkl_filename:

    Returns:
        patches: {layer_id: [[min_x, max_x, min_y, max_y]]}

    """
    patches = {}
    for i, layer in enumerate(target_layers):
        layer_id = layer[0]
        layer_num = layer[1]
        print("Deconvolving patches for layer {}".format(layer_id))
        patches[layer_id] = create_patches_layer(
            image_data, layer_num, layer_id, Dec, save_raw, raw_dir, show)
    return patches


def create_patches_layer(image_data, layer_num, layer_id,
                         Dec, save_raw, raw_dir, show):
    """
    Args:
        image_data:
        layer_num:
        layer_id:
        Dec:
        save_raw:
        raw_dir:
        show:

    Returns:
        patches_layer: list of [min_x, max_x, min_y, max_y] for each neuron
    """
    patches_layer = []
    img_index = [0]  # TODO: allow multiple images?
    X_ori = image_data[img_index]
    i = 0
    while True:
        try:
            X_deconv = Dec.get_deconv(image_data[img_index], layer_id, feat_map=i)
        except IndexError:  # off the end of the layer
            return patches_layer
        try:
            coords, _, _ = crop_images(X_ori, X_deconv)
        except TypeError:
            coords = np.array([0, 0, 0, 0])
        patches_layer.append(coords)
        i += 1


def plot_significants_layer(image_data, layer_num, layer_id, significants,
                            Dec, top_x, save=False, save_dir="",
                            show=False):
    """
    Plot the patches associated with the deconvolution of each patch specified in a particular layer in significants.

    Args:
        image_data:
        layer_num:
        layer_id:
        significants:
        Dec:
        top_x:
        save:
        save_dir:
        show:

    Returns:
        layer_data:
    """
    layer_data = []
    for i in significants[:top_x]:
        filename = "{}_{}".format(layer_id, i)
        layer_data.append(
            (layer_num, i,
                plot_deconv(image_data, Dec, layer_id, i, save=save,
                            save_dir=save_dir, filename=filename, show=show)))
    return layer_data


def plot_deconv(data, Dec, layer_id, feat_map, save=False,
                save_dir="", filename="", show=False):
    """
    Plot original images (or cropped versions) and the deconvolution result
    for images specified in img_index, for the target layer and feat_map
    specified in the arguments

    Args:
        data: (numpy array) the image data with shape: (n_samples, n_channels, img_dim1, img_dim2)
        Dec: (DeconvNet) instance of the DeconvNet class
        layer_id: (str) name of the layer we want to visualise
        feat_map: (int) index of the filter to visualise
        save: (bool) should save the output image
        save_dir: (str) directory in which to save the patch (if save=True)
        filename: (str) filename to save the output image
        show: (bool) should show the new image
    Returns:
        input_map: input cropped to deconv patch
        output_map: deconv cropped to patch
        coords: [y_min, y_max, x_min, x_max] of patch
    """
    img_index = [0]  # TODO: allow multiple images?
    X_ori = data[img_index]
    X_deconv = Dec.get_deconv(data[img_index], layer_id, feat_map=feat_map)
    try:
        coords, arr_deconv, arr_ori = crop_images(X_ori, X_deconv)
    except TypeError:
        coords = np.array([0, 0, 0, 0])

    return None, None, coords
