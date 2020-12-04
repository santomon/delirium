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
import importlib
import pickle

from sklearn.decomposition import PCA

import utility
import delirium_config as config
import tqdm
from NeuralTaskonomy.code.encodingmodel.encoding_model import ridge_cv

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
        full_nn_data_path: str,
        module_name: str,
) -> np.ndarray:
    """
    TODO:doc
    :param stim_list:
    :param nn_data_path: path to where the files are located, currently only supports models, generated from inference/inference.py
    :return:
    """

    #VULNERABLE:
    module = importlib.import_module("inference." + module_name)

    data_: t.List[np.ndarray] = []
    for img_name in tqdm.tqdm(stim_list):
        data_path = os.path.join(full_nn_data_path, module.generate_file_name(img_name))
        data_.append(np.load(data_path, allow_pickle=True).flatten())
    data_: np.ndarray = np.array(data_)
    assert len(data_.shape) == 2, "Error: not all datapoints have the same number of parameters!"
    return data_



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



class EncodingModel:

    def __init__(self,
                 data_path: str,
                 module_name: str,
                 model_name: str,
                 save_path: str,
                 do_pca: bool,
                 fix_testing: bool,
                 BOLD5000_ROI_path: str,
                 BOLD5000_Stimuli_path: str,
                 do_cv: bool,
                 subjects: t.List[int] = [1, 2, 3],
                 TR: t.List[int] = [3, 4]
                 ):

        self.data_path = data_path
        self.module_name = module_name
        self.model_name = model_name
        self.save_path = save_path
        self.do_pca = do_pca
        self.fix_testing = fix_testing
        self.subjects = subjects
        self.TR = TR
        self.do_cv = do_cv

        self.brain_data = [load_brain_data(subject=i, tr=TR) for i in subjects]
        self.stim_lists = load_stim_lists(subjects=subjects)

        self.brain_data, self.stim_lists = eliminate_from_data_by_substr(
            self.brain_data,
            self.stim_lists,
            substr='rep_'
        )   # remove stimuli, that were not shown for the first time

        for substr in config.UNWANTED_IMAGES:
            self.brain_data, self.stim_lists = eliminate_from_data_by_substr(self.brain_data, self.stim_lists, substr)

        utility.inspect(self.brain_data)


    def fit_encoding_model_SSF(self, do_permutation: int):

        for subj, brain_data_single in zip(self.subjects, self.brain_data):

            corrs_array, rsqs_array, cv_array, l_score_array, best_l_array, predictions_array = (
                [],
                [],
                [],
                [],
                [],
                [],
            )


            data = load_nn_data(
                stim_list=self.stim_lists[subj - 1],
                full_nn_data_path=os.path.join(self.data_path, self.module_name, self.model_name),
                module_name=self.module_name
            )

            ridged_brain_data_single = dict()
            for roi in config.ROI_LABELS:
                corrs, *cv_outputs =ridge_cv(X=np.float32(data),
                                                y=np.float32(brain_data_single[roi]),
                                                permute_y= do_permutation,
                                                cv=self.do_cv,
                                                pca = self.do_pca,
                                                fix_testing= self.fix_testing,
                                                split_by_runs=False,
                                                repeat=do_permutation)

                if do_permutation:
                    full_save_path = os.path.join(self.save_path, self.module_name, self.model_name, 'permutations', 'subj{}'.format(subj))
                    if not os.path.isdir(full_save_path):
                        os.makedirs(full_save_path)

                    pickle.dump(
                        corrs,
                        open(
                            os.path.join(full_save_path, "permutation_test_on_test_data_corr{}.p".format(roi)),
                            "wb",
                        ),
                    )

                    pickle.dump(
                        cv_outputs[0],
                        open(
                            os.path.join(full_save_path, "permutation_test_on_test_data_pvalue_{}.p".format(roi)),
                            "wb",
                        ),
                    )
                else:
                    corrs_array.append(corrs)  # append values of all ROIs
                    if len(cv_outputs) > 0:
                        rsqs_array.append(cv_outputs[0])
                        cv_array.append(cv_outputs[1])
                        l_score_array.append(cv_outputs[2])
                        best_l_array.append(cv_outputs[3])
                        predictions_array.append(cv_outputs[4])



            outpath = os.path.join(self.save_path, self.module_name, self.model_name)
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
            full_model_name = self.generate_full_model_name(subj)

            pickle.dump(
                corrs_array, open(os.path.join(outpath, "corr_{}.p".format(full_model_name)), "wb")
            )

            if len(cv_outputs) > 0:  # happens if no permutations
                pickle.dump(
                    cv_outputs[0], open(os.path.join(outpath, "rsq_{}.p".format(full_model_name)), "wb")
                )
                pickle.dump(
                    cv_outputs[1],
                    open(os.path.join(outpath, "cv_score_{}.p".format(full_model_name)), "wb"),
                )
                pickle.dump(
                    cv_outputs[2],
                    open(os.path.join(outpath, "l_score_{}.p".format(full_model_name)), "wb"),
                )
                pickle.dump(
                    cv_outputs[3],
                    open(os.path.join(outpath , "best_l_{}.p".format(full_model_name)), "wb"),
                )

                if self.fix_testing:
                    pickle.dump(
                        cv_outputs[4],
                        open(os.path.join(outpath, "pred_{}.p".format(full_model_name)), "wb"),
                    )


    def generate_full_model_name(self, subj):
        return "subj{}_TR{}_{}_{}_{}".format(subj,
                                             "".join([str(tr) for tr in self.TR]),
                                             "pca" if self.do_pca else "nopca",
                                             "fixtesting" if self.fix_testing else "nofixtesting",
                                             "cv" if self.do_cv else "nocv",
                                             )





def copy_astmt_model(model_dir: str, target_base_dir: str=".") -> t.NoReturn:
    # Deprecated:
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
