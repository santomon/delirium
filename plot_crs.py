import argparse
import os
import pickle

import seaborn as sns
from matplotlib import pyplot as plt

import delirium_config

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
        pass

    def load_corr(self):
        pass

    def load_NT_corr(self, task: str, subj: int):

        _path = os.path.join(delirium_config.NT_PATH,
                     "outputs",
                     "encoding_results",
                     "subj{}".format(subj),
                     "corr_taskrepr_{}__TRavg.p".format(task))

        with open(_path, "rb") as f:
            data = pickle.load(f)

        return data