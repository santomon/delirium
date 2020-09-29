import pytest
import inference.inference
import torch
import inference.template_inference as template_inference


def test_infer_folder():
    pass


def test_infer_all():
    pass


def test_infer_single_image():
    pass



@pytest.mark.useless
def test_torch_device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert isinstance(device, torch.device)



##to learn
##parametrization of tests;
##testing single inference with the help of a parametrized model;






