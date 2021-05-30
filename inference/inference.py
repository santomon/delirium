"""

TODO: doc, testing

'framework' is honestly a bit silly; it does literally nothing, lmao
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


identity = lambda *args: args


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
        model_call: t.Callable[[Tensor], t.Any],
        postprocessor: t.Callable[[t.Any], t.Any],
        saver: t.Callable[[t.Any, str, str], t.NoReturn],
        layer: str
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
    :param layer:
    :return:
    """

    files = os.listdir(data_path)

    for file_ in tqdm.tqdm(files):
        loaded_file = loader(os.path.join(data_path, file_))
        data = preprocessor(loaded_file)
        output = model_call(data, layer)
        postprocessed_output = postprocessor(output)
        saver(postprocessed_output, save_path, file_)


def parse_args() -> argparse.ArgumentParser():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--module", default="torch_inference", type=str,
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

    parser_.add_argument("--save_path", default=delirium_config.NN_DATA_PATH, type=str,
                         help="path, where the results of inference will be saved; if none is provided,"
                            "delirium_config.NN_SAVE_PATH will be used")

    parser_.add_argument("--model", default="all", type=str,
                         help="specify, which model of the module to use; refer to the respective csv files, to get "
                              "valid model names; \n"
                              "'all' can be used, to generate features for all models")

    parser_.add_argument("--layer", default=None, type=str,
                         help="specify a layer name, for which the features should be extracted. "
                              "This parameter is ignored when using 'all' keyword in model selection. Instead the last"
                              "available layer from the backbone will be extracted. \n"
                              "For information on extractable layers, please refer to the respective csv files")

    parser_.add_argument("--skip_existing", default=False, action="store_true",
                         help="if passed, inference will be skipped, if a file with the result name already exists in folder")

    if "inference/inference.py" in sys.argv:
        return parser_.parse_args()
    else:
        return parser_.parse_args("")


if __name__ == "__main__":

    parser = parse_args()

    if parser.dataset == "BOLD5000":
        infer_folder = infer_BOLD5K(subdirectories=delirium_config.BOLD5K_PRES_STIM_SUBDIRECTORIES)(infer_folder)

    module = importlib.import_module(parser.module)


    if parser.model == 'all':
        models = module.viable_models
        if parser.layer is not None:
            print("All models were selected. Specified layer name cannot be guaranteed for all models. Defaulting"
                  "to last layer of the backbone.")
        layer = None
    else:
        models = [parser.model]
        layer = parser.layer



    for model in models:
        module.select_model(model)
        infer_folder(
            parser.data_path,
            parser.save_path,
            module.loader,
            module.preprocessor,
            module.model_call,
            module.postprocessor,
            module.saver,
            layer,

        )

    print("Finished extracting features for all models")