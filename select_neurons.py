from __future__ import print_function
from image_utils import load_or_make_perturbations
from keras import backend as K
from utils import *
import numpy as np
import time


def extract_weight(layer, num, metric):
    """Extracts the top num neurons per sample, across
    the channels in the CNNs weights

    Args:
        layer: string: valid keras layer
        num: int -- how many to extract

    Returns:
        list of top num per layer

    """
    curr = layer.weights[0]
    holder = []
    for ind, t_slice in enumerate(curr.get_value()):
        holder.append([ind, metric(t_slice)])
    s_list = np.asarray(sorted(holder, key=lambda v_pair: v_pair[1], reverse=True))
    return s_list[:num, 0]


def extract_activation(layer, num, metric, min_threshold=0.):
    """ Extracts the top num neurons per sample, across
    the channels in the CNNs activations

    Args:
        layer: string valid keras layer
        num: int - how many to extract
        metric: function

    Returns:
        list of top num
    """
    holder = []
    for ind, t_slice in enumerate(layer[0]):
        if np.mean(t_slice) < min_threshold:
            continue
        holder.append([ind, metric(t_slice)])
    s_list = np.asarray(sorted(holder, key=lambda v_pair: v_pair[1], reverse=True))
    return s_list[:num, 0]


def get_target_layers(model):
    """
    Args:
        model: Keras model to inspect.
    Returns:
        list of [layer_name, layer_index] for all Convlution2d Layers in model
    """
    return [[L.name, i] for i, L in enumerate(model.layers) if "Convolution2D" in str(L)]


def select_weight(model, data, target_layers, top_n, activation_fn, save_deconv=False):
    """ Extract the top neurons based on some function of the weight matrix.

    Args:
        model: Model to use.
        data:
        target_layers:
        top_n:
        activation_fn:
        save_deconv:

    Returns:
        list of list of significants.
    """
    significants = []
    for l_name, i in target_layers:
        L = model.layers[i]
        significants.append([l_name, extract_weight(L, top_n, activation_fn)])

    return significants


def select_activation(model, data, target_layers, top_n, activation_fn, save_deconv=False):
    """ Extract the top neurons based on some function of the activation matrix.

    Args:
        model:
        data:
        target_layers:
        top_n:
        activation_fn:
        save_deconv:

    Returns:
        list of list of significants

    """

    significants = []

    for l_name, i in target_layers:
        img_index = [0]  # np.asarray(user_defined_arr)
        X = data[img_index]
        f = K.function([model.layers[0].input],
                       [model.layers[i].output])

        layer_output = f([X])[0]
        significants.append([l_name, extract_activation(layer_output, top_n, activation_fn)])
    return significants


def select_precision(model, perturbations, target_layers, top_n,
                     save_deconv=False):
    """ Select according to the precision of the activation
    Args:
        model (Keras model): neural network to analyze
        perturbations:
        target_layers:
        top_n: int, number of neurons to select in each layer
        save_deconv:
    Returns:
        list of lists of neurons, post selection

    """
    #np.sum(np.sum(np.sum([[np.abs(j) for j in i]for i in x])))*
    #activation_fn = lambda x: 1.0/np.nanmean([[np.var(j) for j in i] for i in x])
    activation_fn = lambda x: 1.0/np.nanmean(np.var(x, axis=2))
    #activation_fn = lambda x: np.nanmean([[np.var(j) for j in i] for i in x])

    # Find the neurons with activations that correlate most significantly with the output activation
    significants = []
    for l_name, i in target_layers:
        print("Calculating precisions in layer {:d}".format(i), end='\r')
        f = K.function([model.layers[0].input],
                       [model.layers[i].output])
        layer_output = f([perturbations])[0]
        # reshape layer_output to be (1, m, n, p, n_perturbations)
        layer_output_reshaped = np.zeros((
            1, layer_output.shape[1], layer_output.shape[2],
            layer_output.shape[3], layer_output.shape[0]))
        for j in range(len(perturbations)):
            layer_output_reshaped[0, :, :, :, j] = layer_output[j, :, :, :]
        significants.append([l_name, extract_activation(layer_output_reshaped,
                             top_n, activation_fn, min_threshold=1e-1)])
    print("Caculating precisions...\t\tFinished calculating precisions.")

    return significants


def vcorrcoef(X, y):
    """ Fast correlation coefficient calculation for matrix X and vector y. """
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym), axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r


def select_correlation(model, perturbations, target_layers, top_n, save_deconv=False):
    """ Select according to the Activation-Output correlation.
    Args:
        model (Keras model): neural network to analyze
        perturbations (numpy array): the image data, shape: (N1 images, N2 channels, N3 height, N4 width)
        target_layers:
        top_n: int, number of neurons to select in each layer
        save_deconv:

    Returns:
        list of lists of neurons, post selection

    """
    significants = []

    output = np.squeeze(model.predict(perturbations)[:, 0])  # for now, only works on a two-class output
    #activation_fn = lambda x: np.nanmean([[np.abs(np.corrcoef(j, output)) for j in i] for i in x])

    def corr(x):
        return np.nanmean([np.abs(vcorrcoef(I, output)) for I in x])

    for l_name, i in target_layers:
        print("Calculating correlations in layer {:d}".format(i), end='\r')
        f = K.function([model.layers[0].input], [model.layers[i].output])
        layer_output = f([perturbations])[0]

        # reshape layer_output to be (1, m, n, p, n_perturbations)
        layer_output_reshaped = np.zeros((
            1, layer_output.shape[1], layer_output.shape[2],
            layer_output.shape[3], layer_output.shape[0]))
        for j in range(len(perturbations)):
            layer_output_reshaped[0, :, :, :, j] = layer_output[j, :, :, :]
        significants.append([l_name, extract_activation(layer_output_reshaped,
                                                        top_n, corr)])
    print("Calculating correlations...\t\tFinished calculating correlations.")
    return significants


def select_neurons(model, target_layers, data, base_vars, config,
                   activation_cases, load_perturb=False, save_perturb=False,
                   perturb_dir="",):
    """
    Select important neurons.

    Args:
        model: Keras model to run deconvolution on
        target_layers: list of layers to inspect (should be Conv2D)
        data: np array of image data
        activation_case: the index of the type of activation
        base_vars:
        config:

    Returns:
        list of list of neuron indices
    """
    significants = []
    perturbations = load_or_make_perturbations(
        data, load_perturb, save_perturb, perturb_dir)
    for activation_case in activation_cases:
        vars_ = unpack(base_vars, activation_case, config)
        top_n = vars_["top_x"]
        activation_fn = vars_["activation_fn"]
        activation_metric = vars_["activation_metric"]
        print("Activation Function {}".format(activation_metric))
        t = time.time()
        if "weight" in activation_metric:
            significants.append(select_weight(model, data, target_layers, top_n, activation_fn))
        elif "activation" in activation_metric:
            significants.append(select_activation(model, data, target_layers, top_n, activation_fn))
        elif "correlation" in activation_metric:
            significants.append(select_correlation(model, perturbations, target_layers, top_n))
        else:
            significants.append(select_precision(model, perturbations, target_layers, top_n))

        print(" "*80, end='\r')
        print("Took {:.3f} seconds.".format(time.time() - t))
    return significants
