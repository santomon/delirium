import sys
import os
sys.path.append(os.path.abspath("./astmt"))

import typing as t


import numpy as np
import pandas as pd
import gdown
import torch
from PIL import Image

from astmt.experiments.dense_predict.pascal_resnet import config as pascal_config
from astmt.experiments.dense_predict.nyud_resnet import config as nyud_config

from astmt.fblib.util.mypath import Path


device = "cuda" if torch.cuda.is_available() else "cpu"
module_name = 'astmt_inference'
model_params: t.Dict[str, t.Any] = pd.read_json("inference/astmt_model_params.json")

# models marked with SSF have been trained by us and will be attempted to be downloaded from the web
# you can use your own trained models by making an appropriate entry in the model_params dictionary
# the key will of your entry will be considered the "model_name" when running an inference task

SSF_model_urls = {
    'pascal_edge_scratch_SSF': "https://drive.google.com/u/0/uc?id=1dOOIZFJUfiegnfeaG-YdX2JwAOxqRNT9",
    'pascal_semseg_scratch_SSF':"",
    'pascal_normals_scratch_SSF':"",
    'pascal_edge_pretr_SSF':"",
    'pascal_semseg_pretr_SSF':"",
    'pascal_normals_pretr_SSF':"",
    'nyud_edge_scratch_SSF':"",
    'nyud_semseg_scratch_SSF':"",
    'nyud_normals_scratch_SSF':"",
    'nyud_edge_pretr_SSF':"",
    'nyud_semseg_pretr_SSF':"",
    'nyud_normals_pretr_SSF':"",
}

viable_models = model_params.keys()
default_model = 'pascal_edge_scratch_SSF'

model_ = None
currently_selected_model:str = None
cfg = None
transformer = None




def loader(path: str):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32)

def preprocessor(data_):

    tmp = transformer({'image': data_})
    return tmp['image'].to(device)

def model_call(data_):

    return {task: model_.forward(data_, task)[1] for task in model_.tasks} # model.forward return is a tuple with [0] being the output
                                                                        # and [1] the features
                                                                        # features is at this point a 4dim tensor with
                                                                        # 64 entries, last 32 belonging to low level output,
                                                                        # first 32 belonging to high level output

def postprocessor(data_: t.Dict[str, torch.Tensor], compress=True):

    # TODO?: no compression for now

    if compress:
        pass

    return {task: np.float32(features.to('cpu')[0]) for task, features in data_.items()}


def generate_file_name(old_file_name, task):

    return old_file_name.split(".")[0] + '_features' + '_' + task + '.npy'



def saver(data_: t.Dict[str, np.ndarray], path: str, file_name: str) -> t.NoReturn:

    for task in data_.keys():
        full_path = os.path.join(path, module_name, currently_selected_model)
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        np.save(os.path.join(full_path, generate_file_name(file_name, task)), data_[32:])




def select_model(model_name: str):

    global model_, cfg, transformer

    if 'SSF' in model_name:
        _download_SSF_model(model_name)

    parse_string = _create_parse_string(model_params[model_name])

    if 'nyud' in model_name:
        cfg = nyud_config.create_config(parse_string)
        model_ = nyud_config.get_net_resnet(cfg)
        transformer = nyud_config.get_transformations(cfg)
    elif 'pascal' in model_name:
        cfg = pascal_config.create_config(parse_string)
        model_ = pascal_config.get_net_resnet(cfg)
        transformer = pascal_config.get_transformations(cfg)
    else:
        raise NotImplementedError("model name should either contain 'nyud' or 'pascal'")

    model_.to(device)
    model_.eval()


def get_features_by_image_path(path_to_file: str):
    pass


def get_features_by_image_data(img_data: np.ndarray):
    pass


def _download_SSF_model(model_name: str):

    url = SSF_model_urls[model_name]
    save_dir = os.path.join(Path.exp_dir(), "SSF")
    tmp_file_name = os.path.join("tmp", model_name + ".zip")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    if not os.path.isfile(tmp_file_name):
        gdown.download(url, tmp_file_name, quiet=False)
        gdown.extractall(tmp_file_name, save_dir)
    else:
        print(tmp_file_name, "already exists, skipping download")







def _create_parse_string(cfg: t.Dict):

    args = []
    for key, value in cfg.items():
        args.append("--" + key)

        if value != "":
            args.append(str(value))

    return args


select_model(default_model)

