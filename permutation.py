import argparse
import typing as t
import os
import pickle

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import delirium_config
from NeuralTaskonomy.code.util.util import pearson_corr, empirical_p

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


class Permutator():

    def __init__(self, repeats=5000):
        # predictions are not offered for NT
        self.repeats = repeats
        self.data = pd.DataFrame(columns=["module_name", "model_name", "subj", "yhat", "ylabel", "ROI", "Hemisphere", "did_pca"])


    def permute(self):
        grouped = self.data.groupby(list(set(self.data.columns) - set(["yhat", "ylabel"])))
        self.grouped_result = grouped.apply(_permute_single, self.repeats)




    def load_predictions(self, module_name, model_name: str, did_pca: bool, fixed_testing: bool, did_cv: bool, TR: t.List,
                         result_path: str = delirium_config.NN_RESULT_PATH, *fname_spec):
        [self._load_prediction(subj, module_name, model_name, did_pca, fixed_testing, did_cv, TR, result_path, *fname_spec)
         for subj in range(1, 4)]

    def _load_prediction(self, subj: int, module_name: str, model_name: str, did_pca: bool, fixed_testing: bool,
                   did_cv: bool,
                   TR: t.List, result_path=delirium_config.NN_RESULT_PATH, *fname_spec):


        _file_name = self._get_filename(subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
        _path = os.path.join(result_path, module_name, model_name, _file_name)

        with open(_path, "rb") as f:
            _data = pickle.load(f)
        self._append_data(_data, module_name, model_name + ("" if len(fname_spec)==0 else "_"+ "_".join(fname_spec)), subj, did_pca=did_pca)
        return _data

    def _get_filename(self, subj, did_pca, fixed_testing, did_cv, TR, *fname_spec):
        return "pred_subj{}_TR{}_{}_{}_{}{}.p".format(subj,
                                                      "".join([str(tr) for tr in TR]),
                                                      "pca" if did_pca else "nopca",
                                                      "fixtesting" if fixed_testing else "nofixtesting",
                                                      "cv" if did_cv else "nocv",
                                                      "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                                      )

    def _append_data(self, _data, module_name, model_name, subj, did_pca):
        for i, (roi, roi_data) in enumerate(zip(delirium_config.ROI_LABELS, _data)):

            _len = len(roi_data[0])
            _data_dict = {
                "module_name": [module_name for _ in range(_len)],
                "model_name": [model_name for _ in range(_len)],
                "subj": [subj for _ in range(_len)],
                "yhat": [x for x in roi_data[0]],
                "ylabel": [x for x in roi_data[1]],
                "ROI": [roi[2:] for i in range(_len)],
                "Hemisphere": [roi[:2] for _ in range(_len)],
                "did_pca": [did_pca for _ in range(_len)]
            }

            rdata = pd.DataFrame(_data_dict)
            self.data = pd.concat([self.data, rdata])



def _permute_single(data: pd.DataFrame, repeats: int):
    # expected keys are: "yhat", "ylabel"
    # to be used on grouped data

    yhat= np.array(list(data["yhat"]))

    ylabel = np.array(list(data["ylabel"]))
    corrs_dist = []
    original_corrs = [pearsonr(ylabel[:, i], yhat[:, i]) for i in range(ylabel.shape[1])]  #list tuple of corrs and pvalues
    corrs_only = [r[0] for r in original_corrs]

    label_idx = np.arange(ylabel.shape[0])
    for _ in range(repeats):
        np.random.shuffle(label_idx)
        y_test_perm = ylabel[label_idx, :]
        perm_corrs = pearson_corr(y_test_perm, yhat, rowvar=False)
        corrs_dist.append(perm_corrs)
    p = empirical_p(corrs_only, np.array(corrs_dist))

    assert len(p) == ylabel.shape[1], "length of p is not equal to the number of voxels"

    return pd.Series([p], index=["empirical_ps"])


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
