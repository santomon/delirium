"""
inference specific for pretrained models provided by pytorch


known problems:
intermediate_layer_getter fails for InceptionV3, googlenet;
    will fail for anything that uses torch.flatten...
    basically will always happen, if neither backbone nor features exist
checkpoint for 'mnasnet1_3' and 'masnet0_75' exists, but the model does not

models that use the Sequential class (i think) like vgg16 will hab ascending integers as layer_names
"""
import argparse
import typing as t
import os
import numpy as np
import pandas as pd

import torch
from torchvision.models import _utils
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

module_name = 'torch_inference'  # VULNERABLE: last verified 26.11.20

torch_dir = 'pytorch/vision'
default_model = 'deeplabv3_resnet101'
# available_models = torch.hub.list(torch_dir)

viable_models = ['alexnet',
    'deeplabv3_resnet101',
    'deeplabv3_resnet50',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'fcn_resnet101',
    'fcn_resnet50',
    'mobilenet_v2',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn']


model_: torch.nn.Module = None
currently_selected_model: str = None


backbone : torch.nn.Module = None
backbone_layer_keys: t.List[str] = None
backbone_return_layers: t.Dict[str, str] = None

intermediate_layer_getter: torch.nn.Module = None


loader = lambda x: np.asarray(Image.open(x))


def preprocessor(data_):
    """

    :param data_:
    :return:
    """

    # create a transformer, the numeric values can be found on https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # luckily torch models all have the same mean and std
    ])
    return transformer(data_).to(device).unsqueeze(0)


def model_call(data_: torch.Tensor) -> t.Dict:
    with torch.no_grad():
        result = intermediate_layer_getter(data_)
        return result[list(result.keys())[-1]]           # take the last ouput from all generated features


def postprocessor(data_: t.Dict, compress=True) -> np.ndarray:

    try:
        result = data_
        if compress:
            result = torch.nn.AvgPool2d(3)(result).cpu()
            result = np.float16(result).squeeze(0)
        return result
    except TypeError as t:
        print(t)
        print(currently_selected_model, "failed. ")
        print("data_ looks like:" )
        print(data_)
        raise TypeError("stop")


def saver(data_: np.ndarray, path: str, file_name: str) -> t.NoReturn:
    full_path = os.path.join(path, currently_selected_model)
    if not os.path.isdir(full_path):
        os.makedirs(full_path)
    np.save(os.path.join(full_path, module_name, generate_file_name(file_name)), data_)


def generate_file_name(old_file_name: str) -> str:
    """
    for a given image name, generate a respective file name, the output should be saved as;
    e.g.  xd.jpg -> xd_features.npy

    can be used to find the files for the regression part
    """
    return old_file_name.split(".")[0] + '_features' + '.npy'


###########

def select_model(model_name: str) -> t.NoReturn:
    """
    API function for Algonauts; for a given model name, change the model of this module to the selected one;
    to get a list of all viable models, refer to viable_models
    """
    global model_, currently_selected_model, backbone_layer_keys, backbone_return_layers, intermediate_layer_getter
    global backbone
    model_ = torch.hub.load(torch_dir, model_name, pretrained=True)
    model_.to(device)
    model_.eval()
    currently_selected_model = model_name

    if hasattr(model_, "backbone"):
        backbone = model_.backbone
    elif hasattr(model_, 'features'):
        backbone = model_.features
    else:
        backbone = model_
        print("backbone could not be found; defaulting to usual model output")
    backbone.to(device)
    backbone.eval()

    # the names of the layers, that can be extracted from the backbone
    backbone_layer_keys = [name for name, module in backbone.named_children()]

    # preparing the model, that can return all layers specified in backbone_layer_keys
    backbone_return_layers = {layer_name: layer_name for layer_name in backbone_layer_keys}
    intermediate_layer_getter = _utils.IntermediateLayerGetter(backbone,
                                                               backbone_return_layers)  # used to generate all features
    intermediate_layer_getter.to(device)
    intermediate_layer_getter.eval()


def get_features_by_image_path(path_to_file: str):  # -> t.OrderedDict[str, torch.Tensor]
    """
    API function for Algonauts; for a given string, that is a path to an image file, compute a dictionary
    of of outputs of various layers from the encoding model

    all keys can be accessed with backbone_layer_keys

    """

    img = loader(path_to_file)

    with torch.no_grad():
        return get_features_by_image_data(img)


def get_features_by_image_data(img_data: np.ndarray):  #-> t.OrderedDict[str, torch.Tensor]
    """
    API function for Algonauts; for a loaded, not yet preprocessed image;
    image data is expected to be in RGB

    computes a dictionary of outputs for various layers from the encoding model,
    please also refer to the function get_features_by_image_path
    """
    with torch.no_grad():
        img_data = preprocessor(img_data)
        return intermediate_layer_getter(img_data)


########################################


def _update_viable_models():
    """
    used for determining all viable models, returning them as a list
    also generates a csv file that, contains the model names with their respective output keys in forward order
    """
    model_names = []
    result_keys = []
    for model_name in torch.hub.list('pytorch/vision'):
        try:
            print(model_name, ":")
            select_model(model_name)
            result = get_features_by_image_path("./sample.jpg")
            model_names.append(model_name)
            result_keys.append(list(result.keys()))
            print("   no issues found")
        except ValueError:
            print(model_name, "failed to get from torchhub")
        except RuntimeError as t:
            print(t)

    df: pd.DataFrame = pd.DataFrame(list(zip(model_names, result_keys)))
    df.columns = ['model_name', 'output_keys']
    df.to_csv('./inference/torch_model_out_dictkeys.csv')

    return model_names

##############################################

select_default_model = lambda: select_model(default_model)
select_default_model()