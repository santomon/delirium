import argparse
import os
import pickle
import typing as t

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import delirium_config

sns.set_style("whitegrid")


def parse_args():
    _parser = argparse.ArgumentParser()

    _parser.add_argument("--fix_testing", default=False, action="store_true",
                         help="if fixed testing was applied for the models in question")
    _parser.add_argument("--do_pca", default=False, action="store_true",
                         help="if pca was applied for the models in question")
    _parser.add_argument("--data_path", type="str", default=delirium_config.NN_RESULT_PATH,
                         help="baseline location where all the model results")
    _parser.add_argument("--fname_spec", type="str", nargs="*",
                         help="optional args for specifications to find the file name, usually the same ones, that were"
                              "used with main.py, or none at all")



class Plotter:

    def __init__(self):
        self.data = pd.DataFrame(columns = ["module_name", "model_name", "subj", "correlation", "ROI", "Hemisphere", "did_pca"])

    def load_corrs(self, module_name, model_name, did_pca, fixed_testing, did_cv, TR, result_path=delirium_config.NN_RESULT_PATH, *fname_spec):
        return [self._load_corr(subj, module_name, model_name, did_pca, fixed_testing, did_cv, TR, result_path, *fname_spec) for subj in range(1, 4)]


    def load_NT_corrs(self, task: str):
        return [self._load_NT_corr(subj, task) for subj in range(1, 4)]


    def plot_average_for_all_subjects(self, hue_order, plot, hue="model_name", **kwargs):
        if plot == sns.barplot or plot == "bar":
            plot = sns.barplot
        elif plot == sns.violinplot or plot == 'violin':
            plot = sns.violinplot
        elif plot == sns.boxplot or plot =="box":
            plot = sns.boxplot
        else:
            print("plot should be one of bar, violin or box")

        if 'palette' not in kwargs.keys():
            kwargs['palette'] = sns.color_palette("colorblind")

        fig = plt.figure(figsize=((40, 9)))
        ax = fig.add_subplot(1, 1, 1)
        plot(
            x="ROI",
            y="correlation",
            hue=hue,
            hue_order=hue_order,
            data=self.data,
            ax=ax,
            **kwargs
        )
        plt.show()




    def plot_for_all_subjects_individually(self, hue_order, plot, hue="model_name", **kwargs):

        if plot == sns.barplot or plot == "bar":
            plot = sns.barplot
        elif plot == sns.violinplot or plot == 'violin':
            plot = sns.violinplot
        elif plot == sns.boxplot or plot =="box":
            plot = sns.boxplot
        else:
            print("plot should be one of bar, violin or box")

        if 'palette' not in kwargs.keys():
            kwargs['palette'] = sns.color_palette("colorblind")

        grid = sns.FacetGrid(
            self.data,
            row="subj",
            row_order=[1, 2, 3],
            # legend_out=True,
            despine=True,
            height=3,
            aspect=5,
            dropna=False,
        )

        grid.map_dataframe(
            plot,
            "ROI",
            "correlation",
            hue=hue,
            hue_order= hue_order,
            **kwargs
        )

        grid.set_axis_labels("ROI", "Correlations (r)")

        handles = grid._legend_data.values()
        labels = grid._legend_data.keys()
        grid.fig.legend(
            title="Tasks",
            handles=handles,
            labels=labels,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.49, 0.97),
            frameon=False,
        )
        ax = grid.axes
        sns.despine(fig=grid.fig, ax=ax, left=True, bottom=True)

        plt.show()




    def _load_NT_corr(self,subj: int, task: str):

        _path = os.path.join(delirium_config.NT_PATH,
                     "outputs",
                     "encoding_results",
                     "subj{}".format(subj),
                     "corr_taskrepr_{}__TRavg.p".format(task))

        with open(_path, "rb") as f:
            _data = pickle.load(f)

        self._append_data(_data, "NeuralTaskonomy", task, subj, did_pca=False)
        return _data


    def _load_corr(self, subj: int,  module_name: str, model_name: str, did_pca: bool, fixed_testing: bool, did_cv: bool,
                  TR: t.List, result_path= delirium_config.NN_RESULT_PATH,  *fname_spec):

        _file_name = self._get_filename(subj, module_name, model_name, did_pca, fixed_testing, did_cv, TR, result_path, *fname_spec)
        _path = os.path.join(result_path, module_name, model_name, _file_name)

        with open(_path, "rb") as f:
            _data = pickle.load(f)
        self._append_data(_data, module_name, model_name + ("" if len(fname_spec)==0 else "_"+ "_".join(fname_spec)), subj, did_pca=did_pca)
        return _data

    def _append_data(self, _data, module_name, model_name, subj, did_pca):

        for i, corr in enumerate(_data):
            for j, r in enumerate(corr):
                vd = dict()
                vd["correlation"] = r[0]
                vd["ROI"] = delirium_config.ROI_LABELS[i][2:]
                vd["hemisphere"] = delirium_config.ROI_LABELS[i][0:2]
                vd["module_name"] = module_name
                vd["model_name"] = model_name
                vd["subj"] = subj
                vd["did_pca"] = did_pca

                self.data = self.data.append(vd, ignore_index=True)

    def _get_filename(self, subj, module_name, model_name, did_pca, fixed_testing, did_cv, TR, result_path=delirium_config.NN_RESULT_PATH, *fname_spec):
        return "corr_subj{}_TR{}_{}_{}_{}{}.p".format(subj,
                                             "".join([str(tr) for tr in TR]),
                                             "pca" if did_pca else "nopca",
                                             "fixtesting" if fixed_testing else "nofixtesting",
                                             "cv" if did_cv else "nocv",
                                             "" if len(fname_spec) == 0 else "_" + "_".join(fname_spec)
                                             )




