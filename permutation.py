import argparse
import typing as t
import os
import pickle
import itertools


import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from pandas.core.groupby.groupby import GroupBy as pdGroupBy
import seaborn as sns
import matplotlib.pyplot as plt


import delirium_config
from NeuralTaskonomy.code.util.util import pearson_corr, empirical_p
import utility


non_sns_kwargs = ["tick_labels", "backend", "palette", "horizontal_yticks"]
def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


class Permutator():

    def __init__(self, repeats=5000):
        # predictions are not offered for NT
        self.repeats = repeats
        self.data = pd.DataFrame(columns=["module_name", "model_name",
                                          "hemisphere", "ROI", "subj", "did_pca", "fixed_testing",
                                          "did_cv", "TR", "task", "fname_spec", "yhat", "ylabel"])
        self.grouped_result = pd.DataFrame(columns=["module_name", "model_name",
                                          "hemisphere", "ROI", "subj", "did_pca", "fixed_testing",
                                          "did_cv", "TR", "task", "fname_spec", "empirical_ps", "corr_dist", "acc_corrs"])

        self.roiwise_two_stat_ps = dict()


    def permute(self, save_permutations=True, save_dir_root=delirium_config.NN_RESULT_PATH):
        # grouped = self.data.groupby(list(set(self.data.columns) - set(["yhat", "ylabel"])))
        grouped = self.data.groupby(list(self.data.columns[:-2]), sort=False)       # everything except "yhat" and "ylabel" should be keys
                                                                        # naturally assumes "yhat and "ylabel" are last
        self.grouped_result = grouped.apply(_permute_single, self.repeats, save_permutations, save_dir_root).reset_index()



    def load_permutations_and_pvalues(self, module_name, model_name: str, did_pca: bool, fixed_testing: bool, did_cv: bool, TR: t.List,
                         task:str, result_path: str = delirium_config.NN_RESULT_PATH, *fname_spec):

        """
        WIP
        """
        _root = os.path.join(result_path, module_name, model_name)
        for subj in range(1, 4):
            corr_file_name = utility.generate_corr_file_name(subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
            with open(os.path.join(_root, corr_file_name), "rb") as f:
                corrs_raw = pickle.load(f)


            for i, roi in enumerate(delirium_config.ROI_LABELS):

                perm_result_path = os.path.join(_root, "permutation_results")
                perm_file_name = utility.generate_permutation_file_name(roi[:2], roi[2:], subj, did_pca, fixed_testing, did_cv,
                                                                        TR, *fname_spec)
                with open(os.path.join(perm_result_path, perm_file_name), "rb") as f:
                    permutated_corrs = pickle.load(f)

                pvalues_file_name = utility.generate_pvalues_file_name(roi[:2], roi[2:], subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
                with open(os.path.join(perm_result_path, pvalues_file_name), "rb") as f:
                    pvalues = pickle.load(f)

                self._append_result_data(permutated_corrs, corrs_raw[i], pvalues, roi, module_name, model_name, subj,
                                         did_pca, fixed_testing, did_cv, TR, task, *fname_spec)


    def permutation_roiwise_two_stat_p(self, correction="fdr", save=True,
                                       save_name="permutation_riowise_ps.p", save_dir=delirium_config.NN_RESULT_PATH):
        """

        """

        roiwise_groups: pdGroupBy = self.grouped_result.groupby(["ROI", "subj"], sort=False)
        result = dict()

        for group_name, group_roi in roiwise_groups:
            valid_group_keys = list(group_roi.columns[:-3])  # last three are "empirical_ps", "corr_dist", "acc_corr"
            valid_group_keys.remove("hemisphere")  # hemisphere is not part of grouping

            roiwise_result = group_roi.groupby(valid_group_keys, sort=False)

            roiwise_result = groupby_combine(roiwise_result, empirical_two_stat_p)

            if correction == "fdr":
                roiwise_result = matrix_fdrcorrection(roiwise_result)

            result.update({group_name: roiwise_result})
        # self.final_result = roiwise_groups.apply(lambda x: utility.groupby_combine(x, empirical_two_stat_p))

        self.roiwise_two_stat_ps = result

        if save:
            try:
                if save_name[-2:] != ".p":
                    save_name = save_name + ".p"
            except IndexError:
                save_name = save_name + ".p"

            full_save_file = os.path.join(save_dir, save_name)
            with open(full_save_file, "wb") as f:
                pickle.dump(self.roiwise_two_stat_ps, f)


    def roiwise_spearmanr(self, replacement_dict: dict={}, save=True, save_name="roiwise_spearmanrs.p", save_dir=delirium_config.NN_RESULT_PATH):

        def mergeLHRH(df: pd.DataFrame) -> pd.DataFrame:
            cols = list(df.columns)
            cols.remove("hemisphere")
            cols.remove("acc_corrs")
            return df.groupby(cols, sort=False).aggregate(lambda x: list(itertools.chain.from_iterable(x))).drop("hemisphere", axis=1)

        def only_corrs(df: pd.DataFrame) -> pd.DataFrame:
            df["acc_corrs"] = df["acc_corrs"].apply(lambda x: [c for c, p in x])
            return df

        df = self.grouped_result.drop(["empirical_ps", "corr_dist", "fname_spec"], axis=1).replace(replacement_dict)  # drop/rename whatever messes with the grouping
        df = mergeLHRH(df)
        df = df.reset_index()
        df = only_corrs(df)

        roiwise_groups: pdGroupBy = df.groupby(["ROI", "subj"], sort=False)
        result = dict()

        for group_name, group_roi in roiwise_groups:
            roiwise_result = utility.groupby_except(group_roi, ["task", "acc_corrs"], sort=False)
            roiwise_result = groupby_combine(roiwise_result, voxelwise_spearmanr)
            result.update({group_name: roiwise_result})

        self.roiwise_spearmanrs = result

        if save:
            try:
                if save_name[-2:] != ".p":
                    save_name = save_name + ".p"
            except IndexError:
                save_name = save_name + ".p"

            full_save_file = os.path.join(save_dir, save_name)
            with open(full_save_file, "wb") as f:
                pickle.dump(self.roiwise_spearmanrs, f)


    def plot_spearmanr(self, save=True, figname=os.path.join(delirium_config.NN_RESULT_PATH, "spearmanr"), *args, **kwargs):
        """

        """
        if len(self.roiwise_spearmanrs) == 0:
            print("self.roiwise_two_stat_ps is empty, nothing to plot")
            return

        import matplotlib
        import matplotlib.patches as mpatches
        if "backend" in kwargs:
            if kwargs['backend'] == "pgf":
                matplotlib.use("pgf")  # src: https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/ 31.12.2020
                matplotlib.rcParams.update({
                    "pgf.texsystem": "pdflatex",
                    'font.family': 'serif',
                    'text.usetex': True,
                    'pgf.rcfonts': False,
                })
        else:
            matplotlib.use("agg")


        if 'palette' in kwargs.keys():
            palette = sns.color_palette(kwargs['palette'])
        else:
            palette = sns.color_palette("rocket")


        fig, axes = plt.subplots(3, 5)

        for x, (axes_horizontal, subj) in enumerate(zip(axes, range(1, 4))):
            for y, (ax, roi) in enumerate(zip(axes_horizontal, delirium_config.ROI)):

                if 'tick_labels' in kwargs.keys():
                    tick_labels = kwargs['tick_labels']
                else:
                    tick_labels = self.roiwise_spearmanrs[(roi, subj)].columns


                heatmap = sns.heatmap(self.roiwise_spearmanrs[(roi, subj)], vmin=0, vmax=1, ax=ax, linewidth=.5,
                            xticklabels=tick_labels,
                            yticklabels=tick_labels,
                            cbar=True if y == len(axes_horizontal) - 1 else False,
                            cmap=palette,
                            square=True,
                            *args, **_only_sns_kwargs(kwargs))

                if "horizontal_yticks" in kwargs.keys():
                    if kwargs["horizontal_yticks"]:
                        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)

                if x != len(axes) - 1:
                    ax.get_xaxis().set_visible(False)
                else:
                    ax.set_xlabel(roi)

                if y != 0:
                    ax.get_yaxis().set_visible(False)
                else:
                    ax.set_ylabel("subj = {}".format(subj))

        plt.show()
        if save:
            if "backend" in kwargs:
                if kwargs['backend'] == "pgf":
                    fig.set_size_inches(6.30045, fig.get_figheight() * 6.30045 / fig.get_figwidth())
            path = os.path.dirname(figname)
            if not os.path.isdir(path):
                os.makedirs(path)
            fig.savefig(figname, bbox_inches="tight")



    def plot_two_stat_ps(self, plot_alpha: t.Union[bool, float]=False,
                         save=True, figname=os.path.join(delirium_config.NN_RESULT_PATH, "two_stat_ps"), *args, **kwargs):
        """
        tick_labels: custom tick labels to be used, instead of tuples from grouping
        """

        if len(self.roiwise_two_stat_ps) == 0:
            print("self.roiwise_two_stat_ps is empty, nothing to plot")
            return

        import matplotlib
        import matplotlib.patches as mpatches
        if "backend" in kwargs:
            if kwargs['backend'] == "pgf":
                matplotlib.use("pgf")  # src: https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/ 31.12.2020
                matplotlib.rcParams.update({
                    "pgf.texsystem": "pdflatex",
                    'font.family': 'serif',
                    'text.usetex': True,
                    'pgf.rcfonts': False,
                })
        else:
            matplotlib.use("agg")


        if 'palette' in kwargs.keys():
            palette = sns.color_palette(kwargs['palette'])
        else:
            palette = sns.color_palette("rocket")


        fig, axes = plt.subplots(3, 5)

        for x, (axes_horizontal, subj) in enumerate(zip(axes, range(1, 4))):
            for y, (ax, roi) in enumerate(zip(axes_horizontal, delirium_config.ROI)):

                if 'tick_labels' in kwargs.keys():
                    tick_labels = kwargs['tick_labels']
                else:
                    tick_labels = self.roiwise_two_stat_ps[(roi, subj)].columns


                heatmap = sns.heatmap(self.roiwise_two_stat_ps[(roi, subj)] if not plot_alpha else \
                                      self.roiwise_two_stat_ps[(roi, subj)] > plot_alpha
                                      , vmin=0, vmax=1, ax=ax, linewidth=.5,
                            xticklabels=tick_labels,
                            yticklabels=tick_labels,
                            cbar=True if y == len(axes_horizontal) - 1 and not plot_alpha else False,
                            cmap=palette,
                            square=True,
                            *args, **_only_sns_kwargs(kwargs))

                if "horizontal_yticks" in kwargs.keys():
                    if kwargs["horizontal_yticks"]:
                        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)

                if x != len(axes) - 1:
                    ax.get_xaxis().set_visible(False)
                else:
                    ax.set_xlabel(roi)

                if y != 0:
                    ax.get_yaxis().set_visible(False)
                else:
                    ax.set_ylabel("subj = {}".format(subj))

        if plot_alpha:
            greater_than_alpha = mpatches.Patch(color=palette[-1], label="p-value greater than {}".format(plot_alpha))
            less_than_alpha = mpatches.Patch(color=palette[0], label="p-value less than {}".format(plot_alpha))
            fig.legend(handles=[greater_than_alpha, less_than_alpha],
                       loc="upper center",
                       ncol=2,
                       )
            # 0e0e25
            # faebdd

        plt.show()
        if save:
            if "backend" in kwargs:
                if kwargs['backend'] == "pgf":
                    fig.set_size_inches(6.30045, fig.get_figheight() * 6.30045 / fig.get_figwidth())
            path = os.path.dirname(figname)
            if not os.path.isdir(path):
                os.makedirs(path)
            fig.savefig(figname, bbox_inches="tight")





    def load_predictions(self, module_name, model_name: str, did_pca: bool, fixed_testing: bool, did_cv: bool, TR: t.List, task: str,
                         result_path: str = delirium_config.NN_RESULT_PATH, *fname_spec):
        [self._load_prediction(subj, module_name, model_name, did_pca, fixed_testing, did_cv, TR, task, result_path, *fname_spec)
         for subj in range(1, 4)]

    def _load_prediction(self, subj: int, module_name: str, model_name: str, did_pca: bool, fixed_testing: bool,
                   did_cv: bool,
                   TR: t.List, task: str, result_path=delirium_config.NN_RESULT_PATH, *fname_spec):


        _file_name = self._get_filename(subj, did_pca, fixed_testing, did_cv, TR, *fname_spec)
        _path = os.path.join(result_path, module_name, model_name, _file_name)

        with open(_path, "rb") as f:
            _data = pickle.load(f)
        self._append_data(_data, module_name, model_name, subj, did_pca, fixed_testing, did_cv, TR, task,*fname_spec)
        return _data

    def _get_filename(self, subj, did_pca, fixed_testing, did_cv, TR, *fname_spec):
        return "pred_subj{}_TR{}_{}_{}_{}{}.p".format(subj,
                                                      "".join([str(tr) for tr in TR]),
                                                      "pca" if did_pca else "nopca",
                                                      "fixtesting" if fixed_testing else "nofixtesting",
                                                      "cv" if did_cv else "nocv",
                                                      "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                                      )

    def _append_data(self, _data, module_name, model_name, subj, did_pca, fixed_testing, did_cv, TR, task, *fname_spec):
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
                "task": [task for _ in range(_len)],
                "fname_spec": [fname_spec for _ in range(_len)]
            }

            rdata = pd.DataFrame(_data_dict)
            self.data = pd.concat([self.data, rdata])


    def _append_result_data(self, corr_dist,acc_corr, pvalues, roi,
                            module_name, model_name, subj, did_pca, fixed_testing, did_cv, TR, task, *fname_spec):

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
            "task": [task],
            "fname_spec": [fname_spec],
            "empirical_ps": [pvalues],
            "corr_dist": [corr_dist],
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
        permutation_file_name = utility.generate_permutation_file_name(*data.name[2:-2], *data.name[-1])  # -2, instead of 1, so as to leave out the task for generating the file name
        pvalues_file_name = utility.generate_pvalues_file_name(*data.name[2:-2], *data.name[-1])

        pickle.dump(corr_dist, open(os.path.join(full_path, permutation_file_name), "wb"))
        pickle.dump(p, open(os.path.join(full_path, pvalues_file_name), "wb"))

    assert len(p) == ylabel.shape[1], "length of p is not equal to the number of voxels"

    return pd.Series([p, corr_dist, acc_corrs], index=["empirical_ps", "corrs_dist", "acc_corrs"])


def empirical_two_stat_p(group1: pd.DataFrame, group2: pd.DataFrame):
    """
    expects group1 and group2 to be a DataFrame with columns
    "corr_dist" and "acc_corrs"
    only expects 2 rows (1 for each hemisphere), where group{}.loc[i, "corr_dist"] is a 2-dim array
    and                                                group{}.loc[i, "acc_corrs"] is a list of tuples: [(corr, pvalue)]

    computes the empirical p for permutated(group1) - permutated(group2) > actual group1 - actual group2

    the lower the pvlaue, the higher the probability that group1 > group2 should be

    CAVE: using nanmean instead of mean, atm
    """

    group1 = group1.reset_index()
    group2 = group2.reset_index()
    corr_dist1 = np.hstack((group1.loc[0, "corr_dist"], group1.loc[1, "corr_dist"]))
    corr_dist1_mean = np.nanmean(corr_dist1, axis=1)
    acc1 = group1.loc[0, "acc_corrs"] + group1.loc[1, "acc_corrs"]  # concat for LH, and RH
    acc1 = [corr for corr, pvalue in acc1]
    acc1_mean = np.nanmean(acc1)

    corr_dist2 = np.hstack((group2.loc[0, "corr_dist"], group2.loc[1, "corr_dist"]))
    corr_dist2_mean = np.nanmean(corr_dist2, axis=1)
    acc2 = group2.loc[0, "acc_corrs"] + group2.loc[1, "acc_corrs"]
    acc2 = [corr for corr, pvalue in acc2]
    acc2_mean = np.nanmean(acc2)

    p = empirical_p(acc1_mean - acc2_mean, corr_dist1_mean - corr_dist2_mean, dim=1)

    return p

def voxelwise_spearmanr(group1: pd.DataFrame, group2: pd.DataFrame):
    """
    expects group1 and group2 to be a DataFrame with columns
    "acc_corrs"

    computes the mean voxelwise Spearman's correlation between the two groups, based on correlation values for each task
    """

    group1 = group1.reset_index()
    group2 = group2.reset_index()

    corrs1 = np.array(group1["acc_corrs"])
    corrs2 = np.array(group2["acc_corrs"])

    return np.nanmean(mapped_spearmanr(corrs1, corrs2, axis=1))




def mapped_spearmanr(array1: np.ndarray, array2: np.ndarray, axis=0) -> np.ndarray:
    """
    computes only column- or row-wise Spearman correlation for two 2D-arrays with the same shape;
    returns a 1D array with the Spearman correlations

    if axis=0:
    M1 = [[ 1  2  3  4]
         [ 5  6  7  8]
         [ 9 10 11 12]]
    M2 = [[10 11 12 13]
         [14 15 16 17]
         [18 19 20 21]]
    Result = [1. 1. 1. 1.]
    """
    if axis == 1:
        array1 = array1.transpose()
        array2 = array2.transpose()
    assert array1.shape == array2.shape, "array1 is shape {} and array2 is shape {}".format(array1.shape, array2.shape)

    rs = []
    for x, y in zip(array1, array2):

        r, p = spearmanr(x, y)
        rs.append(r)
    return np.array(rs)


def choose_from_triangles(matrix: np.ndarray, f: t.Callable[[t.Any, t.Any], bool]) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    assumes matrix is of shape (i, i) and an np.ndarray;

    given a matrix and f: x, y -> bool
    returns a tuple of np.ndarrays of indices: (a, b) if f(matrix[a, b], matrix[b, a]) == True else (b, a),

    excluding diagonal entries of the matrix;
    e.g.:
    (np.array([a1, a2, b3]), np.array([b1, b2, a3]))
    """

    ut: t.Tuple[np.ndarray, np.ndarray] = np.triu_indices(matrix.shape[0], 1)
    lt: t.Tuple[np.ndarray, np.ndarray] = np.tril_indices(matrix.shape[0], -1)
    # upper triangle and lower triangle are ordered, such that (a, b) and (b, a) are at the same position

    result = (np.zeros(ut[0].shape[0], dtype=np.int), np.zeros(ut[0].shape[0], dtype=np.int))
    for i, (a, b) in enumerate(zip(matrix[ut], matrix[lt])):
        if f(a, b):
            result[0][i] = ut[0][i]
            result[1][i] = ut[1][i]
        else:
            result[0][i] = lt[0][i]
            result[1][i] = lt[1][i]
    return result


def matrix_fdrcorrection(matrix: t.Union[np.ndarray, pd.DataFrame]) -> t.Union[np.ndarray, pd.DataFrame]:
    """
    assumes matrix is of shape (i, i);
    assumes for non-diagonal p-values: matrix[a,b] = 1 - matrix[b, a];
    applies fdrcorrection using the smaller p-values of matrix[a,b] and matrix[b,a] for all a,b
    the larger p-values in matrix[a, b] are overwritten with 1 - newpvalue(matrix[b,a])
    """

    matrix_ = np.array(matrix)  # cast into np.ndarray

    small_pvalue_indices: t.Tuple[np.ndarray, np.ndarray] = choose_from_triangles(matrix_, lambda x, y: x < y)
    large_pvalue_indices: t.Tuple[np.ndarray, np.ndarray] = (small_pvalue_indices[1], small_pvalue_indices[0])
    _, new_small_pvalues = fdrcorrection(matrix_[small_pvalue_indices])

    matrix_[small_pvalue_indices] = new_small_pvalues
    matrix_[large_pvalue_indices] = 1 - new_small_pvalues
    matrix[:] = matrix_  # if original form was pd.DataFrame, entries will be safely overwritten
    return matrix



def groupby_combine(groupby_dataframe: pdGroupBy,
                    func: t.Callable[[pd.DataFrame, pd.DataFrame, t.Any], t.Any], *args, **kwargs) -> pd.DataFrame:

    """
    combines the groups of a pandas groupby object with a function, that uses 2 groups as arguments, such that
    if the groups are x,y,z and function is f:
         x         y        z
    x: f(x,x)    f(x,y)   f(x,z)
    y: f(y,x)    f(y,y)   f(y,z)
    z: f(z,x)    f(z,y)   f(z,z)

    """

    _len = len(groupby_dataframe)

    _group_names = [name for name, group in groupby_dataframe]
    _empty = np.zeros((_len, _len))
    result = pd.DataFrame(_empty)
    result.columns = pd.MultiIndex.from_tuples(_group_names, )
    result.index = pd.MultiIndex.from_tuples(_group_names)

    combinations = itertools.product(groupby_dataframe, repeat=2)

    for ((group1_name, group1_df), (group2_name, group2_df)) in combinations:
        partial_result = func(group1_df, group2_df, *args, **kwargs)

        result.loc[group1_name, group2_name] = partial_result

    return result


def _only_sns_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if key not in non_sns_kwargs}

def main():
    args = parse_args()


if __name__ == '__main__':
    main()

    x: pdGroupBy
    x.first

