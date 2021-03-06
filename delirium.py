"""
contains functions an customization
that are specifically tailored towards the BOLD5000 data set and NeuralTaskonomy

expects you to have cloned https://github.com/ariaaay/NeuralTaskonomy.git somewhere
and edited the variable config.NT_PATH:
"""

import os
import typing as t
import scipy.io
import numpy as np
import sys

import utility
import delirium_config as config
import tqdm

sys.path.append(os.path.join(config.NT_PATH, "code"))

# import encodingmodel.encoding_model


HEMISPHERES = ["LH", "RH"]
ROI = ["OPA", "PPA", "LOC", "EarlyVis", "RSC"]
ROI_LABELS = tuple(hs + roi for hs in HEMISPHERES for roi in ROI)

TO_ELIMINATE = ["golfcourse7.jpg", "childsroom7.jpg", "COCO_train2014_000000000625.jpg"]

brain_dtype = t.List[t.Dict[str, np.ndarray]]

DEFINED_ASTMT_MODELS = ["pascal_resnet", "pascal_mnet", "nyud_resnet"]


def load_brain_data(
        brain_data_path: str = config.BOLD5K_ROI_DATA_PATH,
        subject: int = 1,
        tr: t.Union[int, t.List[int]] = 3,  # int or list of ints
        aggregator: t.Callable[[t.List[np.ndarray]], np.ndarray] = lambda x: np.mean(x, axis=0),
        roi_labels: t.Iterable = ROI_LABELS
) -> t.Dict[str, np.ndarray]:
    """
    TODO:
    doc on params

    :return: dictionary with
        {keys: ROI of type string,
        element: brain_data as an np.ndarray}
    """
    if isinstance(tr, int):
        data_: t.Dict = scipy.io.loadmat(
            os.path.join(brain_data_path, "CSI{}/mat/CSI{}_ROIs_TR{}.mat".format(subject, subject, tr))
        )
        data_.pop('__header__', None)
        data_.pop('__version__', None)
        data_.pop('__globals__', None)
        assert set(data_.keys()) == set(roi_labels), \
            "ROIs in data don't match ROIs in config in CSI{}_ROIs_TR{}.mat".format(subject, tr)

        return data_
    elif isinstance(tr, list):
        tmp_data = [load_brain_data(brain_data_path, subject, tr_) for tr_ in tr]
        return utility.aggregate_over_list_of_dict(tmp_data, aggregator)
    else:
        raise TypeError("tr should be either instance of int or list[int]!")


def load_stim_lists(brain_data_path: str = config.BOLD5K_ROI_DATA_PATH, subjects: t.List[int] = (1, 2, 3)) -> \
    t.List[t.List[str]]:
    stim_lists = []

    for subject in subjects:
        with open(os.path.join(brain_data_path, "stim_lists", "CSI0{}_stim_lists.txt".format(subject))) as f:
            stim_list = f.readlines()
        stim_lists.append([item.strip("\n") for item in stim_list])
    return stim_lists


def eliminate_from_data_by_substr(data_: brain_dtype, stim_lists: t.List[t.List[str]], substr: str = "rep_") -> \
        t.Tuple[brain_dtype, t.List[t.List[str]]]:
    """
    TODO: doc
    :param data_:
    :param stim_lists:
    :param substr:
    :return:
    """

    new_stim_lists: t.List[t.List[str]] = []
    indices: t.List[t.List[int]] = []

    for stim_list in stim_lists:
        tmp_stim_list, tmp_indices = utility.eliminate_by_substr(stim_list, substr)
        new_stim_lists.append(tmp_stim_list)
        indices.append(tmp_indices)

    new_data = []
    for i, data_point in enumerate(data_):
        tmp_data = dict()
        for roi in data_point.keys():
            tmp_data.update({roi: utility.eliminate_by_indices(data_point[roi], indices[i])})
        new_data.append(tmp_data)

    return new_data, new_stim_lists




def load_nn_data(
        stim_list: t.List[str],
        nn_data_path: str = config.NN_DATA_PATH,
        prefix: str = config.NAME_PREFIX,
        suffix: str = config.NAME_SUFFIX,
        file_ending: str = config.NAME_FENDING
) -> np.ndarray:
    """
    TODO:doc
    :param stim_list:
    :param nn_data_path:
    :param prefix:
    :param suffix:
    :param file_ending:
    :return:
    """

    if file_ending in ["npy"]:
        data_: t.List[np.ndarray] = []
        for img_name in tqdm.tqdm(stim_list):
            data_path = os.path.join(nn_data_path, prefix + img_name.split(".")[0] + suffix + "." + file_ending,)
            data_.append(np.load(data_path, allow_pickle=True).flatten())
        data_: np.ndarray = np.array(data_)
        assert len(data_.shape) == 2, "Error: not all datapoints have the same number of parameters!"
        return data_
    else:
        raise NotImplementedError("operation is currently only supported for npy-files")


def get_BOLD5K_Stimuli(target_dir: str=".", chunk_size= 1024*1024*10) -> t.NoReturn:

    import utility
    utility.download_and_extract(config.BOLD5K_STIMULI_URL, "BOLD5K.zip", target_dir, chunk_size=chunk_size)


def get_BOLD5K_ROI_data(target_dir: str=".", chunk_size= 1024*1024*10) -> t.NoReturn:
    import utility
    utility.download_and_extract(config.BOLD5K_ROI_DATA_URL, "ROIs.zip", target_dir, chunk_size=chunk_size)



def rearrange_nn_data(nn_data: np.ndarray,
                      curr_subj: int,
                      next_subj: int,
                      stim_lists: t.List[t.List[str]]):

    idx_tmp = []
    for fname in stim_lists[next_subj - 1]:
        idx_tmp.append(stim_lists[curr_subj - 1].index(fname))

    return nn_data[idx_tmp]



def copy_astmt_model(model_dir: str, target_base_dir: str=".") -> t.NoReturn:

    # TODO: error on gdrive, when too many files; which can happen, if results have already been computed

    for model_name in DEFINED_ASTMT_MODELS:
        if model_name in model_dir:
            base_model = model_name
            break
    else:
        raise ValueError("no astmt model that fits the selected directory")

    base_index = model_dir.index(base_model)
    target_dir = os.path.join(target_base_dir, model_dir[base_index:])
    print(target_dir)

    import shutil

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    if os.path.isdir(target_dir):
        os.rmdir(target_dir)
    shutil.copytree(model_dir, target_dir)


if __name__ == "__main__":
    print("not dead")
