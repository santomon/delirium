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






loader = Image.open


def preprocessor(data_: torch.Tensor):
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
    result = np.float16(result)
    return result


def saver(data_: np.ndarray, path: str, file_name: str) -> t.NoReturn:
    if not os.path.isdir(path):
        os.makedirs(path)
    np.save(os.path.join(path, file_name.split(".")[0] + '_bb_compressed' + '.npy'), data_)