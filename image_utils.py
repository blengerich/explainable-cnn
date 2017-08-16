from __future__ import print_function
from file_io import *
import numpy as np
import os
from PIL import Image, ImageDraw
from utils import *


def load_perturbations(directory):
    # TODO: Implement
    return None


def make_perturbations(data, n_perturbations=50, noise_level=1e-1, save=False, save_dir=""):
    """ Make perturbations of the data and maybe save them.
    Args:
        data: np array of images
        n_perturbations: int, number of perturbations to Create
        noise_level: float, std of normal distribution to create noise
        save: bool, whether to save the perturbations
        save_dir: str, directory in which to save the perturbations
    Return: np Array of images
    """
    print("Generating {} perturbations with noise N(0, {})...".format(
        n_perturbations, noise_level), end='\t')
    perturbations = np.repeat(data, n_perturbations, axis=0)
    for i in range(n_perturbations):
        perturbations[i] *= 1 + i*np.random.normal(
            0.0, noise_level, size=data.shape[1:4])

    print("Finished generating perturbations.")
    if save is True and save_dir is not "":
        print("Saving perturbations to {}".format(save_dir))
        for i, p in enumerate(perturbations):
            im = Image.fromarray(p.T, mode="RGB")
            im.save("{}/perturbation_{}.jpg".format(save_dir, i))

    return perturbations


def load_or_make_perturbations(data, load_perturb, save_perturb, perturb_dir):
    """ Try to load the perturbations, or create them if loading fails.

    Args:
        data: np array of images
        load_perturb: bool, whether or not we should attempt to load from file.
        save_perturb: bool, whether or not we should save to file.
        perturb_dir: str, directory containing the perturbations.

    Returns:
        np array of perturbed images
    """
    if load_perturb:
        if perturb_dir is not "":
            perturbs = load_perturbations(perturb_dir)
        if perturbs is not None:
            return perturbs
        print("Must input perturbation directory to load files. Falling back to creating them.")
    return make_perturbations(data, save=save_perturb, save_dir=perturb_dir)


def process_data(im_name, width, height):
    """
    Open and resize the input stored at im_name.

    Args:
        im_name:
        width:
        height:

    Returns:
        PIL image, float original width, float original height

    """
    im = Image.open(im_name)
    (o_width, o_height) = im.size
    im = im.resize((width, height))
    return im, o_width, o_height


def draw_boxes_layer(layer_num, all_coords, selected_indices, im):
    """
    Draw box outlining the neuron patch.

    Args:
        layer_num: int
        selected_coords: list of coordinates to be drawn
        im: PIL Image to be drawn on

    Returns:
        None
    """
    if layer_num <= 2 or layer_num >= 10:
        return
    else:
        if layer_num <= 4:
            color = "Red"
        elif layer_num >= 5 and layer_num <= 7:
            color = "Green"
        else:
            color = "Blue"
        for i, coord in enumerate(all_coords):
            if i not in selected_indices:
                continue
            [y_min, y_max, x_min, x_max] = coord
            draw = ImageDraw.Draw(im)
            draw.line((x_min, y_min, x_min, y_max), fill=color)
            draw.line((x_min, y_max, x_max, y_max), fill=color)
            draw.line((x_max, y_max, x_max, y_min), fill=color)
            draw.line((x_min, y_min, x_max, y_min), fill=color)


def draw_boxes(coords, selected_neurons, img_dir, picture_name, width, height,
               output_path):
    """ Draw boxes outlining the corresponding patches for the selected neurons.

    Args:
        patches: list of list of coords
        selected_neurons: list of list of ints
        img_dir:
        picture_name:
        width:
        height:
        output_path:

    Returns:
        None
    """
    filename = "{}/{}_0.png".format(get_input_filename(img_dir, picture_name),
                                    picture_name)
    im, o_width, o_height = process_data(filename, width, height)
    make_dir(output_path)

    # Draw All on One
    for i, layer in enumerate(selected_neurons):
        draw_boxes_layer(i, coords[layer[0]], layer[1], im)
    im = im.resize((o_width, o_height))
    print("Saving to {}/All_on_one.jpg".format(output_path))
    im.save("{}/All_on_one.jpg".format(output_path))


def rgb_to_gbr(im):
    """ Rearrange the indices to change RGB to GBR.

    Args:
        im

    Returns:
        Rearranged np array
    """
    im_copy = im.copy()
    im[:, :, 0] = im_copy[:, :, 2]
    im[:, :, 1] = im_copy[:, :, 1]
    im[:, :, 2] = im_copy[:, :, 0]
    return im


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            # Convert to uint to make it look like an image indeed
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros(
                (out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype='uint8'
                    if output_pixel_vals else out_array.dtype
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(
            out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            X[tile_row * tile_shape[1] + tile_col].reshape(
                                img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] +
                                     tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array


def _load_images(base_dir, list_img, height, width, picture_name):
    """Takes in the images as a list, resizes them, and modifies them
    to be appropriate for use with VGG

    Args:
       list_img iterator: images stored in an iterator

    Returns:
        np array: processed images

    """
    data = []
    for im_name in list_img:
        try:
            im_name = '/'.join([base_dir, im_name])
            im = np.asarray(Image.open(im_name).resize(
                (height, width)), dtype=np.float32)
            img_len = len(im.shape)
        except:
            continue

        if img_len < 3:
            continue
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        data.append(im)
    data = np.array(data)
    return data


def load_images(picture_name, height, width, img_dir):
    """Returns the loaded images to make_patch_images

    Args:
        picture_name (string):  which picture_name to load image from
        # Must be present in Data/Img folder
        height (int): height
        width (int): width
        img_dir: str, directory in which the images are located
    Returns:
        loaded images

    """
    dir_name = "{}/{}/".format(img_dir, picture_name)
    print("Loading images from {}".format(dir_name), end='\t')
    list_img = os.listdir(dir_name)
    assert len(list_img) > 0, "Put some images in {}".format(dir_name)
    data = _load_images(dir_name, list_img, height, width, picture_name)
    print("Successfully loaded {} images.".format(len(data)))
    return data



def crop_images(X_ori, X_deconv):
    """
    Args:
        X_ori:
        X_deconv:

    Returns:
        [y_min, y_max, x_min, x_max], arr_deconv, arr_ori
    """
    max_delta_x = 0
    max_delta_y = 0

    # To crop images:
    # First loop to get the largest image size required (bounding box)
    for k in range(X_deconv.shape[0]):
        arr = np.argwhere(np.max(X_deconv[k], axis=0))
        try:
            (ystart, xstart), (ystop, xstop) = arr.min(0), arr.max(0) + 1
        except ValueError:
            print("Encountered a dead filter, retry with different img/filter")
            return
        delta_x = xstop - xstart
        delta_y = ystop - ystart
        if delta_x > max_delta_x:
            max_delta_x = delta_x
        if delta_y > max_delta_y:
            max_delta_y = delta_y

    list_deconv = []
    list_ori = []

    # Then loop to crop all images to the same size
    for k in range(X_deconv.shape[0]):
        arr = np.argwhere(np.max(X_deconv[k], axis=0))
        try:
            (ystart, xstart), (ystop, xstop) = arr.min(0), arr.max(0) + 1
        except ValueError:
            print("Encountered a dead filter, retry with different img/filter")
            return
        # Specific case to avoid array boundary issues
        y_min, y_max = ystart, ystart + max_delta_y
        if y_max >= X_deconv[k].shape[-2]:
            y_min = y_min - (y_max - X_deconv[k].shape[-2])
            y_max = X_deconv[k].shape[-2]

        x_min, x_max = xstart, xstart + max_delta_x
        if x_max >= X_deconv[k].shape[-1]:
            x_min = x_min - (x_max - X_deconv[k].shape[-1])
            x_max = X_deconv[k].shape[-1]

        # Store the images in the dict
        arr_deconv = X_deconv[k, :, y_min: y_max, x_min: x_max]
        arr_ori = X_ori[k, :, y_min: y_max, x_min: x_max]

        list_ori.append(arr_ori)
        list_deconv.append(arr_deconv)
    coords = [y_min, y_max, x_min, x_max]
    arr_deconv = np.array(list_deconv)
    arr_ori = np.array(list_ori)

    return coords, arr_deconv, arr_ori
