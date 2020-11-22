"""
inference specific for deeplabv3
"""
import argparse
import typing as t
import os
import numpy as np
import torch
from torchvision.models import _utils
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

segmentation_models = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']


model_: torch.nn.Module = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
currently_selected_model = 'deeplabv3_resnet101'

backbone = model_.backbone
backbone.to(device)
backbone.eval()


#the names of the layers, that can be extracted from the backbone
backbone_layer_keys = backbone.keys()


#preparing the model, that can return all layers specified in backbone_layer_keys
backbone_return_layers = {layer_name: layer_name for layer_name in backbone_layer_keys}
intermediate_layer_getter = _utils.IntermediateLayerGetter(backbone, backbone_return_layers)  # used to generate all features
intermediate_layer_getter.to(device)
intermediate_layer_getter.eval()





loader = lambda x: np.asarray(Image.open(x))


def preprocessor(data_):
    """

    :param data_:
    :return:
    """

    # create a transformer, the numeric values can be found on https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformer(data_).to(device).unsqueeze(0)


def model_call(data_: torch.Tensor) -> t.Dict:
    with torch.no_grad():
        return backbone(data_)


def postprocessor(data_: t.Dict) -> np.ndarray:

    result = data_['out']
    result = torch.nn.AvgPool2d(3)(result).cpu()
    result = np.float16(result).squeeze(0)
    return result


def saver(data_: np.ndarray, path: str, file_name: str) -> t.NoReturn:
    if not os.path.isdir(path):
        os.makedirs(path)
    np.save(os.path.join(path, file_name.split(".")[0] + '_bb_compressed' + '.npy'), data_)
    


def select_model(model_name: str) -> t.NoReturn:
    global model_, currently_selected_model
    model_ = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    currently_selected_model = model_name


def get_features_by_image_path(path_to_file: str):  # -> t.OrderedDict[str, torch.Tensor]
    """
    API function for Algonauts; for a given string, that is a path to an image file, compute a dictionary
    of of outputs of various layers from the encoding model

    in this case, a resnet-101
    all keys can be accessed with backbone_layer_keys

    the implementation of the resnet-101 in torch is as follows:
    source: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py      #last time verified: 22.11.20
            def _forward_impl(self, x: Tensor) -> Tensor:
                # See note [TorchScript super()]
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)         // is cut off in the deeplabv3 model
                x = torch.flatten(x, 1)     // is cut off in the deeplabv3 model
                x = self.fc(x)              // is cut off in the deeplabv3 model
                return x
    architecture graph: https://www.researchgate.net/figure/The-network-architectures-a-original-deep-ResNet-101-and-b-our-improved-deep_fig1_337010913
    # last time checked: 22.11.20
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



