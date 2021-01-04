import visualpriors
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import os
import numpy as np
import typing as t

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
module_name = "taskonomy_vp_inference"

viable_models = "autoencoding depth_euclidean jigsaw reshading colorization " \
                "edge_occlusion keypoints2d room_layout curvature edge_texture " \
                "keypoints3d segment_unsup2d class_object egomotion nonfixated_pose " \
                "segment_unsup25d class_scene fixated_pose normal segment_semantic " \
                "denoising inpainting point_matching vanishing_point".split(" ")  # slightly different names than in Taskonomy

currently_selected_model:str = None




def select_model(model_name):
    global currently_selected_model
    if model_name in viable_models:
        currently_selected_model = model_name
    else:
        raise NotImplementedError("a task of this name is not available; refer to list_of_tasks for a list of viable tasks")


def loader(path_to_image: str):
    return Image.open(path_to_image)


def preprocessor(pil_image):
    x = TF.to_tensor(TF.resize(pil_image, 256)) * 2 - 1
    return x.unsqueeze_(0).to(device)


def model_call(preprocessed_img, layer):

    # layer is isgnored for now; cant be selected yet; will always be encoder.output with size: [8, 16,16]
    return visualpriors.representation_transform(preprocessed_img, currently_selected_model, device=device)


def postprocessor(result, compress=True):

    return result.to("cpu").squeeze(0).numpy()


def saver(data_: np.ndarray, path: str, file_name: str) -> t.NoReturn:
    full_path = os.path.join(path, module_name, currently_selected_model)
    if not os.path.isdir(full_path):
        os.makedirs(full_path)
    np.save(os.path.join(full_path, generate_file_name(file_name)), data_)




def generate_file_name(old_file_name: str, *fname_spec) -> str:
    """
    for a given image name, generate a respective file name, the output should be saved as;
    e.g.  xd.jpg -> xd_features.npy

    can be used to find the files for the regression part
    """
    return old_file_name.split(".")[0] + '_features' + '.npy'