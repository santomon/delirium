"""
TODO: module docstring
"""
import argparse
import sys
import typing as t
import os
import numpy as np

import delirium_config
import delirium


def parse_args() -> argparse.Namespace:
    """
    TODO: function docstring
    """

    _parser = argparse.ArgumentParser()

    _parser.add_argument("--model", default="all", type=str,
                         help="choose a model for which to run an experiment; "
                              " the string will be treated as a subfolder of the data_path;\n\n"
                              "the keyword 'all' can be used instead, to run a given experiment on all subfolders in "
                              "data_path, that can be acquired with os.listdir")

    _parser.add_argument("--data_path", default=delirium_config.NN_DATA_PATH, type=str,
                         help="base path for, where the features are stored; defaults to delirium_config.NN_DATA_PATH,"
                              "if not specified")
    _parser.add_argument("--save_path", default=delirium_config.NN_RESULT_PATH, type=str,
                         help="path, to where the results will be stored. a subdirectory with the modelname will also"
                              "be created")
    _parser.add_argument("--do_permutations", default=0, type=int,
                         help="if selected, the the regression model will be done the specified number of times "
                              "with randomised train-test-splits; and stored in: \n"
                              "<path>/<model>/permutations")

    _parser.add_argument("--do_pca", action="store_true",
                         help="specify, if PCA should be applied on the data, before running a given experiment;\n"
                              "this is highly recommended, as feature spaces can be really large and potentially cause"
                              "your RAM to explode")
    _parser.add_argument()


    if 'main.py' in sys.argv:
        return _parser.parse_args()
    else:
        return _parser.parse_args("")

def main():
    args: argparse.Namespace = parse_args()

    if args.model == 'all':
        models: t.List[str] = [file for file in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, file))]
    else:
        models: t.List[str] = [os.path.join(args.data_path, args.model)]

    for model in models:
        if args.do_permutations > 0:
            delirium.do





if __name__ == '__main__':
    main()