import os


def make_dir(dirname):
    """ Makes the directory and all containing directories.

    Args:
        dirname: the path of the directory to construct

    Returns:
        None
    """
    print("Making directory {}".format(dirname))
    try:
        os.mkdir(dirname)
    except OSError, err:
        if err.errno == 2:  # No such file or directory
            if '/' in dirname:
                parts = dirname.strip().split("/")
                make_dir('/'.join(parts[:-1]))
                try:
                    os.mkdir(dirname)
                except OSError, err:
                    if err.errno != 17:  # exists
                        raise OSError


def get_input_filename(img_dir, picture_name):
    return '/'.join([os.getcwd(), img_dir, picture_name])


def get_patches_dir(vars_):
    return vars_["pkl_path"]
    #return "{}{}{}.pkl".format(vars_["pkl_path"], vars_["weight"], vars_["picture_name"])


def get_patches_filename(vars_):
    make_dir(vars_["pkl_path"])
    return vars_["pkl_path"] + "/" + "patches.pkl"
    #return "{}{}{}.pkl".format(vars_["pkl_path"], vars_["weight"], vars_["picture_name"])


def get_predicted_prob_filename(vars_):
    return "writeup/One_Predicted_probabilities_{}{}{}.txt".format(
        vars_["weight"], vars_["picture_name"], vars_["activation_metric"])
