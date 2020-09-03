import os
import typing
import torch.nn as nn
import scipy.io
import numpy as np

BOLD5KSUBDIRS = ["COCO", "ImageNet", "Scene"]
FENDINGS = ["jpg", "JPEG", "JPG"]
HEMISPHERES = ["LH", "RH"]
ROI = ["OPA", "PPA", "LOC", "EarlyVis", "RSC"]

ROI_LABELS = [side + roi for side in HEMISPHERES for roi in ROI]


def infer_recursive():
    """
      infers all
    """
    pass


def infer_single_folder(
        img_dir: str,
        out_dir: str,
        fendings: typing.List[str] = FENDINGS,
        model_=None,
        model_params=[],
        compressor=None,
        compress_params=[],
):
    """
      given a list of file endings; applies the model for all those files
      in a specified directory
    """



def load_brain_data(
        brain_data_path: str,
        subject: int,
        tr: typing.Union[int, typing.List[int]],  # int or list of ints
        aggregator: typing.Callable[[typing.List[np.ndarray]], np.ndarray] = lambda x: np.mean(x, axis=0)
) -> typing.Dict[str, np.ndarray]:
    """

    :return: dictionary with
        {keys: ROI of type string,
        element: brain_data as an np.ndarray}
    """
    if isinstance(tr, int):
        data_: typing.Dict = scipy.io.loadmat(
            os.path.join(brain_data_path, "CSI{}/mat/CSI{}_ROIs_TR{}.mat".format(subject, subject, tr))
        )
        data_.pop('__header__', None)
        data_.pop('__version__', None)
        data_.pop('__globals__', None)
        assert set(data_.keys()) == set(ROI_LABELS), \
            "ROIs in data don't match ROIs in config in CSI{}_ROIs_TR{}.mat".format(subject, tr)
        return data_
    elif isinstance(tr, list):
        tmp_data = [load_brain_data(brain_data_path, subject, tr_) for tr_ in tr]
        return aggregate_over_list_of_dict(tmp_data, aggregator)
    else:
        raise TypeError("tr should be either instance of int or list[int]!")


def aggregate_over_list_of_dict(data_: typing.List[typing.Dict[str, np.ndarray]],
                                aggregator: typing.Callable[[typing.List[np.ndarray]], np.ndarray]
                                ) -> typing.Dict[str, np.ndarray]:
    """
    :param data_: list of brain data dictionaries with the following structure: [{roi (string): brain data (np.ndarray}]
    :param aggregator: an aggregation function that is usable on typing.List[np.ndarray]
    :return: an aggregated dictionary of the data
    """
    assert all(
        [data_point.keys() == data_[0].keys() for data_point in data_]), "dictionaries in list don't match in keys!"

    result_data: typing.Dict[str, np.ndarray] = dict()
    for key in data_[0].keys():
        result_data.update({key: aggregator([data_point[key] for data_point in data_])})
    return result_data

