import argparse
import typing as t
import os
import pickle

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from statsmodels.stats.multitest import fdrcorrection


import delirium_config
from NeuralTaskonomy.code.util.util import pearson_corr, empirical_p

import utility


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


class Permutator():

    def __init__(self, repeats=5000):
        # predictions are not offered for NT
        self.repeats = repeats
        self.data = pd.DataFrame(columns=["module_name", "model_name",
                                          "hemisphere", "ROI", "subj", "did_pca", "fixed_testing",
                                          "did_cv", "TR", "fname_spec", "yhat", "ylabel"])
        self.grouped_result = pd.DataFrame(columns=["module_name", "model_name",
                                          "hemisphere", "ROI", "subj", "did_pca", "fixed_testing",
                                          "did_cv", "TR", "fname_spec", "empirical_ps", "corr_dist", "acc_corrs"])


    def permute(self, save_permutations=True, save_dir_root=delirium_config.NN_RESULT_PATH):
        # grouped = self.data.groupby(list(set(self.data.columns) - set(["yhat", "ylabel"])))
        grouped = self.data.groupby(list(self.data.columns[:-2]))       # everything except "yhat" and "ylabel" should be keys
                                                                        # naturally assumes "yhat and "ylabel" are last
        self.grouped_result = grouped.apply(_permute_single, self.repeats, save_permutations, save_dir_root)


    def load_permutations_and_pvalues(self, module_name, model_name: str, did_pca: bool, fixed_testing: bool, did_cv: bool, TR: t.List,
                         result_path: str = delirium_config.NN_RESULT_PATH, *fname_spec):

        """
        WIP
        """
        for subj in range(1, 4):
            corr_file_name = utility.generate_corr_file_name(subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
            with open(os.path.join(module_name, model_name, corr_file_name), "rb") as f:
                corrs_raw = pickle.load(f)


            for i, roi in enumerate(delirium_config.ROI_LABELS):

                perm_result_path = os.path.join(result_path, module_name, model_name, "permutation_results")
                perm_file_name = utility.generate_permutation_file_name(roi[:2], roi[2:], subj, did_pca, fixed_testing, did_cv,
                                                                        TR, *fname_spec)
                with open(os.path.join(perm_result_path, perm_file_name), "rb") as f:
                    permutated_corrs = pickle.load(f)

                pvalues_file_name = utility.generate_pvalues_file_name(roi[:2], roi[2:], subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
                with open(os.path.join(perm_result_path, pvalues_file_name), "rb") as f:
                    pvalues = pickle.load(f)

                self._append_result_data()


    def permutation_roiwise_two_stat_p(self):
        """

        """







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
        self._append_data(_data, module_name, model_name, subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
        return _data

    def _get_filename(self, subj, did_pca, fixed_testing, did_cv, TR, *fname_spec):
        return "pred_subj{}_TR{}_{}_{}_{}{}.p".format(subj,
                                                      "".join([str(tr) for tr in TR]),
                                                      "pca" if did_pca else "nopca",
                                                      "fixtesting" if fixed_testing else "nofixtesting",
                                                      "cv" if did_cv else "nocv",
                                                      "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                                      )

    def _append_data(self, _data, module_name, model_name, subj, did_pca, fixed_testing, did_cv, TR, *fname_spec):
        for i, (roi, roi_data) in enumerate(zip(delirium_config.ROI_LABELS, _data)):

            _len = len(roi_data[0])
            _data_dict = {
                "module_name": [module_name for _ in range(_len)],
                "model_name": [model_name for _ in range(_len)],
                "subj": [subj for _ in range(_len)],
                "yhat": [x for x in roi_data[0]],
                "ylabel": [x for x in roi_data[1]],
                "ROI": [roi[2:] for _ in range(_len)],  # one of OPA, PPA, LOC, EarlyVis or RHC
                "hemisphere": [roi[:2] for _ in range(_len)],  # either LH or RH
                "did_pca": [did_pca for _ in range(_len)],
                "did_cv": [did_cv for _ in range(_len)],
                "fixed_testing": [fixed_testing for _ in range(_len)],
                "TR": [tuple(TR) for _ in range(_len)],
                "fname_spec": [fname_spec for _ in range(_len)]
            }

            rdata = pd.DataFrame(_data_dict)
            self.data = pd.concat([self.data, rdata])


    def _append_result_data(self, pvalues, corr_dist, acc_corr, roi,
                            module_name, model_name, subj, did_pca, fixed_testing, did_cv, TR, *fname_spec):

        _data_dict = {
            "module_name": [module_name],
            "model_name": [model_name],
            "subj": [subj],
            "ROI": [roi[2:]],  # one of OPA, PPA, LOC, EarlyVis or RHC
            "hemisphere": [roi[:2]],  # either LH or RH
            "did_pca": [did_pca],
            "did_cv": [did_cv],
            "fixed_testing": [fixed_testing],
            "TR": [tuple(TR)],
            "fname_spec": [fname_spec],
            "empirical_ps": [pvalues],
            "permutated_corrs": [corr_dist],
            "acc_corrs": [acc_corr],
        }

        rdata = pd.DataFrame(_data_dict)
        self.grouped_result = pd.concat([self.grouped_result, rdata])



def _permute_single(data: pd.DataFrame, repeats: int, save_permutations, save_dir_root):
    """
    expected keys are: "yhat", "ylabel"
    "module_name" and "model_name" are expected to be the first 2 entries of data.name;
    last entry of data.name should be a tuple of any additional keywords that are to be included in the save name;
    this is the case, when they are the first 2 columns of the original data frame;

    to be used on grouped data from Permutator:

    permutes predictions and labels and computes correlation and pvalue
    """

    yhat= np.array(list(data["yhat"]))

    ylabel = np.array(list(data["ylabel"]))
    corr_dist = []
    acc_corrs = [pearsonr(ylabel[:, i], yhat[:, i]) for i in range(ylabel.shape[1])]  #list tuple of corrs and pvalues
    corrs_only = [r[0] for r in acc_corrs]

    label_idx = np.arange(ylabel.shape[0])
    for _ in range(repeats):
        np.random.shuffle(label_idx)
        y_test_perm = ylabel[label_idx, :]
        perm_corrs = pearson_corr(y_test_perm, yhat, rowvar=False)
        corr_dist.append(perm_corrs)
    p = empirical_p(corrs_only, np.array(corr_dist))


    if save_permutations:
        full_path = os.path.join(save_dir_root, data.name[0], data.name[1], "permutation_results")
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        permutation_file_name = utility.generate_permutation_file_name(*data.name[2:-1], *data.name[-1])
        pvalues_file_name = utility.generate_pvalues_file_name(*data.name[2:-1], *data.name[-1])

        pickle.dump(corr_dist, open(os.path.join(full_path, permutation_file_name), "wb"))
        pickle.dump(p, open(os.path.join(full_path, pvalues_file_name), "wb"))

    assert len(p) == ylabel.shape[1], "length of p is not equal to the number of voxels"

    return pd.Series([p, corr_dist, acc_corrs], index=["empirical_ps", "corrs_dist", "acc_corrs"])


def empirical_two_stat_p(group1: pd.DataFrame, group2: pd.DataFrame, correction="fdr"):
    """
    expects group1 and group2 to be a DataFrame with columns
    "corr_dist" and "acc_corrs"
    only expects 2 rows (1 for each hemisphere), where group{}.loc[i, "corr_dist"] is a 2-dim array
    and                                                group{}.loc[i, "acc_corrs"] is a list of tuples: [(corr, pvalue)]

    computes the empirical p for permutated(group1) - permutated(group2) > actual group1 - actual group2

    the lower the pvlaue, the higher the probability that group1 > group2 should be
    """

    corr_dist1 = np.hstack((group1.loc[0, "corr_dist"], group1.loc[1, "corr_dist"]))
    acc1 = group1.loc[0, "acc_corrs"] +  group1.loc[1, "acc_corrs"]  # concat
    acc1 = [corr for corr, pvalue in acc1]
    acc1_mean = np.mean(acc1)

    corr_dist2 = np.hstack((group2.loc[0, "corr_dist"], group2.loc[1, "corr_dist"]))
    acc2 = group2.loc[0, "acc_corrs"] +  group1.loc[2, "acc_corrs"]
    acc2 = [corr for corr, pvalue in acc2]
    acc2_mean = np.mean(acc2)

    p = empirical_p(acc1_mean - acc2_mean, corr_dist1 - corr_dist2)

    if correction == "fdr":
        p = fdrcorrection(p)[1]
    return p









def main():
    args = parse_args()


if __name__ == '__main__':
    main()
