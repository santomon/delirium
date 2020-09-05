"""

TODO: doc, testing
"""



import typing as t
import tqdm
import torch
import tensorflow as tf
import os

import delirium_config
import inference.deeplabv3_inference

Tensor = t.Union[torch.Tensor, tf.Tensor]
Type_of_Data = t.Any


#decorator function to use
def infer_BOLD5K(subdirectories=delirium_config.BOLD5K_PRES_STIM_SUBDIRECTORIES):
    """

    :param subdirectories:
    :return:
    """

    def inner(infer_function):

        def wrapper(*args, **kwargs):

            parent_path = kwargs['path']
            for subdirectory in subdirectories:

                kwargs['path'] = os.path.join(parent_path, subdirectory)
                infer_function(*args, **kwargs)
        return wrapper
    return inner



@infer_BOLD5K(subdirectories=delirium_config.BOLD5K_PRES_STIM_SUBDIRECTORIES)
def infer_folder(
        data_path: str,
        save_path: str,
        loader: t.Callable[[str], Type_of_Data],
        preprocessor: t.Callable[[Type_of_Data], Tensor],
        mode_call: t.Callable[[Tensor], t.Any],
        postprocessor: t.Callable[[t.Any], t.Any],
        saver: t.Callable[[t.Any, str, str], t.NoReturn],
) -> t.NoReturn:
    """
    TODO:doc
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







if __name__ == "__main__":

    infer_folder(
        os.path.join(delirium_config.BOLD5K_STIMULI_PATH, delirium_config.BOLD5K_PRES_STIM_SUBPATH),
        delirium_config.NN_SAVE_PATH,
        inference.deeplabv3_inference.loader,
        inference.deeplabv3_inference.preprocessor,
        inference.deeplabv3_inference.model_call,
        inference.deeplabv3_inference.postprocessor,
        inference.deeplabv3_inference.saver
    )