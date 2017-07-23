from __future__ import print_function
import analysis
from deconv import load_or_create_patches
from select_neurons import get_target_layers, select_neurons
from image_utils import load_images, draw_boxes
import json
import KerasDeconv
import numpy as np
from utils import *
import file_io
#from components.component_predictor import run_predictor, reconstruct_data
np.set_printoptions(threshold=np.inf)


def crop(image, coords):
    """ Crops an image (np array) to selected coordinates.
    Args:
        image: np array
        coords: [x_min, x_max, y_min, y_max]
    Returns:
        np array of the same size as image, but cropped.
    """
    cropped = np.zeros_like(image)
    cropped[coords[0]:coords[1], coords[2]:coords[3], :] = image[coords[0]:coords[1], coords[2]:coords[3], :]
    return cropped


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_train_test_sets(image_data, target_layers, all_coords, selected_indices, labels, p_train=0.6):
    """
    Args:
        image_data: list of np arrays, each entry is an image
        all_coords: list (pictures) of dict {layer_name: [list of coords [x_min, x_max, y_min, y_max]]}
        selected_indices: list of list of indices
        labels:
        p_train:

    Returns:
        np array X_train, np array X_test, np array Y_train, np array Y_test
    """
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for pic_index, image in enumerate(image_data):
        for j, (layer_name, layer_index) in enumerate(target_layers):
            [l_name, layer_neurons] = selected_indices[pic_index][j]
            assert(l_name == layer_name)
            for neuron_index in layer_neurons:
                coords = all_coords[pic_index][layer_name][int(neuron_index)]
                if np.random.uniform() < p_train:
                    X_train.append(crop(image, coords))
                    Y_train.append(labels[pic_index])
                else:
                    X_test.append(crop(image, coords))
                    Y_test.append(labels[pic_index])

    X_train, Y_train = unison_shuffled_copies(
        np.array(X_train), np.array(Y_train))
    X_test, Y_test = unison_shuffled_copies(
        np.array(X_test), np.array(Y_test))
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)

    overlaps = []
    accuracies = []
    base_vars = {}

    for weight_file in get_weight_files(config):
        print("Weight File: {}".format(weight_file))
        overlaps_weight = []
        patches_weight = []
        selected_weight = []
        picture_labels = []
        image_datas = []
        base_vars["weight"] = weight_file
        model = load_model(config["height"], config["width"],
                           "{}/{}.hdf5".format(config["model_dir"], weight_file))
        target_layers = get_target_layers(model)
        Dec = KerasDeconv.DeconvNet(model)
        for picture_name, picture_label in get_picture_names(config):
            print("Picture Name: {}".format(picture_name))
            base_vars["picture_name"] = picture_name
            base_vars = unpack(base_vars, "", config)
            make_dir(base_vars["pkl_path"])

            # Load or Create Patches
            print("Loading or creating Patches...")
            image_data = load_images(picture_name, config["height"],
                                     config["width"], base_vars["img_dir"])
            image_datas.extend(image_data)
            patches = load_or_create_patches(
                image_data, target_layers, Dec,
                save_pkl=True, pkl_dir=file_io.get_patches_dir(base_vars),
                save_raw=False)
            patches_weight.append(patches)
            picture_labels.append(picture_label)

            # Select Neurons
            print("Selecting Neurons...")
            selected = select_neurons(
                model, target_layers, image_data, base_vars, config,
                activation_cases=base_vars["activation_cases"])
            selected_weight.append(selected)

            if picture_label == 1:
                # Draw Boxes
                print("Drawing Boxes")
                for activation_case, selected_metric in zip(base_vars["activation_cases"], selected):
                    vars_ = unpack(base_vars, activation_case, config)
                    draw_boxes(
                        patches, selected_metric, vars_["img_dir"],
                        picture_name, vars_["width"],
                        vars_["height"], vars_['bounding_box_path'])

                # Calculate the Overlap between the selections by the
                # Activation-Output Correlation metric and the Precision metric
                print("Calculating Overlap")
                overlap = analysis.calc_overlap(selected[4], selected[5])
                overlaps_weight.append(overlap)
                print("Overlap Size: {}".format(overlap))
        overlaps.append(overlaps_weight)

        # Calculate the accuracy for each selection metric
        print("Calculating Accuracies")
        accuracies_weight = []
        for i in range(len(selected_weight[0])):
            X_train, X_test, Y_train, Y_test = get_train_test_sets(
                image_datas, target_layers, patches_weight,
                [s[i] for s in selected_weight], picture_labels)
            [my_loss, my_accuracy] = analysis.calc_accuracy(
                X_train, X_test, Y_train, Y_test)
            accuracies_weight.append(my_accuracy)
        print("Accuracies: {}".format(accuracies_weight))
        accuracies.append(accuracies_weight)

    print("Overlaps: {}".format(overlaps))
    print("Accuracies: {}".format(accuracies))
