"""
contains functions that are mostly generic

"""


import os
import typing as t
import scipy.io
import numpy as np

import pandas as pd
from pandas.core.groupby.groupby import GroupBy as pdGroupBy
import itertools



FENDINGS = ("jpg", "JPEG", "JPG")


def infer_recursive():
    """
      infers all
    """
    pass


def infer_single_folder(
        img_path: str,
        out_path: str,
        fendings: t.List[str] = FENDINGS,
        model_=None,
        model_params=(),
        compressor=None,
        compress_params=(),
):
    """
      given a list of file endings; applies the model for all those files
      in a specified directory
    """






def aggregate_over_list_of_dict(data_: t.List[t.Dict[str, np.ndarray]],
                                aggregator: t.Callable[[t.List[np.ndarray]], np.ndarray]
                                ) -> t.Dict[str, np.ndarray]:
    """
    :param data_: list of brain data dictionaries with the following structure: [{roi (string): brain data (np.ndarray}]
    :param aggregator: an aggregation function that is usable on t.List[np.ndarray]
    :return: an aggregated dictionary of the data
    """
    assert all(
        [data_point.keys() == data_[0].keys() for data_point in data_]), "dictionaries in list don't match in keys!"

    result_data: t.Dict[str, np.ndarray] = dict()
    for key in data_[0].keys():
        result_data.update({key: aggregator([data_point[key] for data_point in data_])})
    return result_data


def inspect(data_: t.Union[t.List[t.Dict[str, t.Any]], t.Dict[str, t.Any]]) -> t.NoReturn:
    """
    TODO:doc
    :param data_: data that is to be inspected; for any
    prints out more information on the data
    """
    if isinstance(data_, dict):
        if hasattr(list(data_.values())[0], "shape"):
            for roi, data in data_.items():
                print(roi, data.shape)
    elif isinstance(data_, list):
        for elem in data_:
            inspect(elem)
            print()


def eliminate_by_substr(data_: t.List[str], substr: str) -> t.Tuple[t.List[str], t.List[int]]:
    """
    :param data_: a list of strings
    :param substr: substr as filter parameter
    :return: returns the list without strings that contained the substring and a list indices they had in the original
    """
    if isinstance(data_, list) and all([isinstance(data_point, str) for data_point in data_]):
        indices = [i for i, string in enumerate(data_) if substr in string]
        list_ = [string for string in data_ if substr not in string]
        return list_, indices
    else:
        raise NotImplementedError("Error: Currently this function can only be used on a list that contains only strings")


def eliminate_by_indices(data_: t.Union[t.List, np.ndarray],
                           indices: t.List[int]) -> t.Union[t.List, np.ndarray]:
    """
    TODO: doc
    :param data_:
    :param indices:
    :return:
    """
    if isinstance(data_, np.ndarray):
        return np.delete(data_, indices, 0)
    elif isinstance(data_, list):
        return [data_point for i, data_point in enumerate(data_) if i not in indices]

    else:
        raise NotImplementedError("Currently not supported for other types than np.ndarrays and lists")


def download_and_extract(url: str, tmp_name:str, target_dir:str =".", chunk_size=1024*1024*10) ->t.NoReturn:
    import requests
    bold5k_data = requests.get(url, allow_redirects=True, stream=True)

    with open(os.path.join(target_dir, tmp_name), "wb") as fd:

        try:
            import tqdm

            for chunk in tqdm.tqdm(bold5k_data.iter_content(chunk_size=chunk_size)):
                fd.write(chunk)
        except ModuleNotFoundError:
            for chunk in bold5k_data.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    import zipfile
    bold5k = zipfile.ZipFile(os.path.join(target_dir, tmp_name))
    bold5k.extractall(target_dir)


def identity(x: t.Any) -> t.Any:
    return x


def groupby_except(df: pd.DataFrame, notby: t.Iterable, *args, **kwargs):
    """
    groups the dataframe by all columns, except the ones in notby;
    will fail if not-to-group-bys cannot be found in df.columns;
    args, kwargs will be passed to df.groupby
    """
    cols = list(df.columns)
    for col in notby:
        cols.remove(col)
    return df.groupby(cols, *args, **kwargs)



# certain file names:
def generate_permutation_file_name(hemisphere: str, roi: str, subj: int,  did_pca, fix_testing, did_cv, TR: t.List, *fname_spec):
    return "perm_{}{}_subj{}_TR{}_{}_{}_{}{}.p".format(
                                    hemisphere,
                                    roi,
                                    subj,
                                    "".join([str(tr) for tr in TR]),
                                    "pca" if did_pca else "nopca",
                                    "fixtesting" if fix_testing else "nofixtesting",
                                    "cv" if did_cv else "nocv",
                                    "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                    )


def generate_pvalues_file_name(hemisphere: str, roi: str, subj: int, did_pca, fix_testing, did_cv,  TR: t.List, *fname_spec):
    return "pvalues_{}{}_subj{}_TR{}_{}_{}_{}{}.p".format(
                                    hemisphere,
                                    roi,
                                    subj,
                                    "".join([str(tr) for tr in TR]),
                                    "pca" if did_pca else "nopca",
                                    "fixtesting" if fix_testing else "nofixtesting",
                                    "cv" if did_cv else "nocv",
                                    "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                    )


def generate_corr_file_name(subj: int, did_pca, fix_testing, did_cv,  TR: t.List, *fname_spec):
    return "corr_subj{}_TR{}_{}_{}_{}{}.p".format(
                                    subj,
                                    "".join([str(tr) for tr in TR]),
                                    "pca" if did_pca else "nopca",
                                    "fixtesting" if fix_testing else "nofixtesting",
                                    "cv" if did_cv else "nocv",
                                    "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                    )