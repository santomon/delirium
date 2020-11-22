"""
inference specific for deeplabv3
"""
import argparse
import typing as t
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model_: torch.nn.Module = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
backbone = model_.backbone
backbone.to(device)
backbone.eval()



backbone_layer_keys = ["conv1", "bn1", "relu", "maxpool",
                       "layer1", "layer2", "layer3", "layer4",
                       "avgpool", "flatten", "fc"]


loader = np.asarray(Image.open)


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



def get_features_by_image_path(path_to_file: str) -> t.Dict[str, torch.Tensor]:
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

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
    architecture graph: https://www.researchgate.net/figure/The-network-architectures-a-original-deep-ResNet-101-and-b-our-improved-deep_fig1_337010913
    # last time checked: 22.11.20
    """

    img = loader(path_to_file)
    img = preprocessor(img)


