import numpy as np
import os
import typing as t
import torch


device: str = "cuda" if torch.cuda.is_available() else "cpu"

def loader(data_):
    pass


def preprocessor(data_):
    pass


def model_call(data_):
    pass


def postprpocessor(data_):
    pass


def saver(data_, save_path, name):
    pass