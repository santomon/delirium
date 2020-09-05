"""

TODO: doc, testing
"""

import argparse
import typing as t
import tqdm
import torch
import tensorflow as tf
import os
import importlib

import sys

sys.path.append(".")

import delirium_config

Tensor = t.Union[torch.Tensor, tf.Tensor]
Type_of_Data = t.Any


# decorator function to use
def infer_BOLD5K(subdirectories=delirium_config.BOLD5K_PRES_STIM_SUBDIRECTORIES):
    """
    Why go this far for so little?
        -Why indeed
    :param subdirectories:
    :return:
    """

    def inner(infer_function):
        def wrapper(*args, **kwargs):
            parent_path = args[0]
            args_ = list(args)
            for subdirectory in subdirectories:
                try:
                    args_[0] = os.path.join(parent_path, subdirectory)
                except:
                    try:
                        kwargs['data_path'] = os.path.join(parent_path, subdirectory)
                    except:
                        pass
                infer_function(*args_, **kwargs)
        return wrapper
    return inner


def infer_folder(
        data_path: str,
        save_path: str,
        loader: t.Callable[[str], Type_of_Data],
        preprocessor: t.Callable[[Type_of_Data], Tensor],
        mode_call: t.Callable[[Tensor], t.Any],
        postprocessor: t.Callable[[t.Any], t.Any],
        saver: t.Callable[[t.Any, str, str], t.NoReturn],
) -> t.Any:
    """
    TODO:doc
        "upgrade", to incorporate a namegenerator; would be helpful to check, if file should be skipped

    :param data_path:
    :param save_path:
    :param loader:
    :param preprocessor:
    :param mode_call:
    :param postprocessor:
    :param saver:
    :return:
    """

    files = os.listdir(data_path)

    for file_ in tqdm.tqdm(files):
        loaded_file = loader(os.path.join(data_path, file_))
        data = preprocessor(loaded_file)
        output = mode_call(data)
        postprocessed_output = postprocessor(output)
        saver(postprocessed_output, save_path, file_)


def parse_args() -> argparse.ArgumentParser():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--model_module", default="deeplabv3_inference", type=str,
                         help="will attempt to import model_module"
                             "any model_module has to implement methods:"
                             "      data_path: str,"
                             "     save_path: str,"
                             "       loader: t.Callable[[str], Type_of_Data],"
                             "       preprocessor: t.Callable[[Type_of_Data], Tensor],"
                             "       mode_call: t.Callable[[Tensor], t.Any],"
                             "      postprocessor: t.Callable[[t.Any], t.Any],"
                             "       saver: t.Callable[[t.Any, str, str], t.NoReturn],"
                             "specify a file which implements method, to run network inference on a set of data"
                             "depending on the implementation of the model by the original authors, this can vary from"
                             "easy to rather complicated;"
                         )

    parser_.add_argument("--dataset", default="BOLD5000", type=str,
                         help="essentially specify, if the dataset is used for the BOLD5000 dataset")

    parser_.add_argument("--data_path",
                         default=os.path.join(delirium_config.BOLD5K_STIMULI_PATH, delirium_config.BOLD5K_PRES_STIM_SUBPATH),
                         type=str,
                         help="directory of the dataset; if none is provided, delirium_config specification will be used")

    parser_.add_argument("--save_path", default=delirium_config.NN_SAVE_PATH, type=str,
                        help="path, where the results of inference will be saved; if none is provided,"
                            "delirium_config.NN_SAVE_PATH will be used")

    if "inference/inference.py" in sys.argv:
        return parser_.parse_args()
    else:
        return parser_.parse_args("")


if __name__ == "__main__":

    parser = parse_args()

    if parser.dataset == "BOLD5000":
        infer_folder = infer_BOLD5K(subdirectories=delirium_config.BOLD5K_PRES_STIM_SUBDIRECTORIES)(infer_folder)

    model_ = importlib.import_module(parser.model_module)

    infer_folder(
        parser.data_path,
        parser.save_path,
        model_.loader,
        model_.preprocessor,
        model_.model_call,
        model_.postprocessor,
        model_.saver
    )