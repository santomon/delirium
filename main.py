"""
TODO: module docstring
"""
import argparse
import sys
import typing as t
import os
import numpy as np


import sys
sys.path.append('NeuralTaskonomy/code')
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
                              "<data_path>/<module>/..., that can be acquired with os.listdir")

    _parser.add_argument("--module", default="all", type=str,
                         help="select the module, which the model was part of. if 'all' keyword is used,"
                              " will try to run the experiment on all subfolders of data_path")

    _parser.add_argument("--data_path", default=delirium_config.NN_DATA_PATH, type=str,
                         help="base path for, where the features are stored; defaults to delirium_config.NN_DATA_PATH,"
                              "if not specified")
    _parser.add_argument("--save_path", default=delirium_config.NN_RESULT_PATH, type=str,
                         help="path, to where the results will be stored. a subdirectory with the modelname will also"
                              "be created")
    _parser.add_argument("--do_permutations", default=0, type=int,
                         help="if selected, the the regression model will be done the specified number of times "
                              "with randomised train-test-splits; and stored in: \n"
                              "<save_path>/<module>/<model>/permutations")

    _parser.add_argument("--do_pca", action="store_true",
                         help="specify, if PCA should be applied on the data, before running a given experiment;\n"
                              "this is highly recommended, as feature spaces can be really large and potentially cause"
                              "your RAM to explode")
    _parser.add_argument("--BOLD5000_ROI_path", default=os.path.join(delirium_config.BOLD5K_ROI_DATA_PATH), type=str,
                         help="specify, where to find the BOLD5000 ROI dataset")

    _parser.add_argument("--BOLD5000_Stimuli_path", default=os.path.join(delirium_config.BOLD5K_STIMULI_PATH, delirium_config.BOLD5K_PRES_STIM_SUBPATH),
                         type=str,
                         help="specify, where to find the presented stimuli from the BOLD5000 experiment")


    if 'main.py' in sys.argv:
        return _parser.parse_args()
    else:
        return _parser.parse_args("")


def main():
    args: argparse.Namespace = parse_args()

    if args.module == 'all':
        module_names = os.listdir(args.data_path)
    else:
        module_names = [args.module]

    for module_name in module_names:

        print("module_name")

        if args.model == 'all':
            model_names: t.List[str] = [file for file in os.listdir(os.path.join(args.data_path, module_name)) if os.path.isdir(os.path.join(args.data_path, module_name, file))]
        else:
            model_names: t.List[str] = [args.model] if os.path.isdir(os.path.join(args.data_path, module_name, args.model)) else []

        if model_names == []:
            print("model(s) could not be found!")

        for model_name in model_names:
            if args.do_permutations > 0:
                delirium.permutation_test_SSF( n = args.do_permutations,
                                           data_path = args.data_path,
                                           module_name = module_name,
                                           model_name = model_name,
                                           save_path = args.save_path,
                                           do_pca = args.do_pca,
                                           BOLD5000_ROI_path = args.BOLD5000_ROI_path,
                                           BOLD5000_Stimuli_path = args.BOLD5000_Stimuli_path,
                )





if __name__ == '__main__':
    main()