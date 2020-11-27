"""
the correct order for the keys in the output of the features:
res{number} -> ascending order, followed by
p{number} -> descending order

e.g.
[res2, res3, re4, p6, p5, p4, p3, p2]

known issue:
some models only return one layer; more specifically networks, where the head only needs one output from the backbone
not exactly sure how to change that without injections into detectron2 itself

maybe just ignore these models for now; fpn-backbone models are safe to use; alternatively, check
d2_model_out_dictkeys.csv
"""

import numpy as np
import os
import typing as t
import torch
import pandas as pd
import collections

from PIL import Image

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.structures import ImageList

from detectron2.model_zoo import model_zoo
from detectron2.modeling import meta_arch

module_name = "detectron_inference"

viable_models = model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()
# the viable model names can be found there
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py  # last checked 23.11.20



default_model = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"



device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FeatureExtractor:
    """

    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take RGB image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg_path: str):

        self.cfg = model_zoo.get_config(cfg_path, trained=True)
        self.model = model_zoo.get(cfg_path, trained=True)
        self.model.eval()
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.cfg.INPUT.FORMAT == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            images = self.preprocess_image([inputs])

            predictions = self.model.backbone(images.tensor)
            if hasattr(self.model.backbone, "bottom_up"):  # fpn backbones are actually 2 backbones
                predictions_bottom_up = self.model.backbone.bottom_up(images.tensor)
                predictions_bottom_up.update(predictions)
                return predictions_bottom_up
            else:
                return predictions


    def preprocess_image(self, batched_inputs):
        # all models from detectron2 preprocess the images the same way
        # this could change in the future; fingers crossed that fbresearch stays consistent
        # reference: https://github.com/facebookresearch/detectron2/tree/master/detectron2/modeling/meta_arch  # last checked: 23.11.20

        images = [x["image"].to(self.model.device) for x in batched_inputs]
        images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        return images



predictor: FeatureExtractor = FeatureExtractor(default_model)
currently_selected_model = default_model


def loader(path_to_image: str) -> np.ndarray:

    img = np.asarray(Image.open(path_to_image))  # loaded as RGB
    return img


def preprocessor(data_):
    #identity; incorporated in modelcall
    return data_


def model_call(data_):
    result = _reorder_features(predictor(data_))
    return result[list(result.keys())[-1]]


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
    full_path = os.path.join(path, module_name, currently_selected_model)
    if not os.path.isdir(full_path):
        os.makedirs(full_path)
    np.save(os.path.join(full_path, generate_file_name(file_name)), data_)


def generate_file_name(old_file_name: str) -> str:
    """
    for a given image name, generate a respective file name, the output should be saved as;
    e.g.  xd.jpg -> xd_features.npy

    can be used to find the files for the regression part
    """
    return old_file_name.split(".")[0] + '_features' + '.npy'


def select_model(model_name: str):
    """
    change the model, that will be used by the predictor;
    to get the model_names, refer to viable models at the top of this file
    """
    global predictor, currently_selected_model
    predictor = FeatureExtractor(model_name)
    currently_selected_model = model_name


def get_features_by_image_path(path_to_image: str) -> t.Dict[str, torch.Tensor]:
    """
    API function for Algonauts; for a given string, that is a path to an image file, compute a dictionary
    of of outputs of various layers from the encoding model
    """
    image_data = loader(path_to_image)
    with torch.no_grad():
        return get_features_by_image_data(image_data)


def get_features_by_image_data(image_data: np.ndarray) -> t.Dict[str, torch.Tensor]:
    """
    API function for Algonauts; for a loaded, not yet preprocessed image,
    return outputs of various layers of the encoding model;
    image data is expected to be in RGB

    the layers here are provided by detectron2 itself;
    no IntermediateLayerGetter
    """
    with torch.no_grad():
        return _reorder_features(predictor(image_data))


def _reorder_features(outputs: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
    """
    vulnarable to future updates;
    last reviewed: 23.11.20

    reoorders the features based on the heuristic described at the top of this file
    and turns the dict into an ordered dict (type annotation for t.OrderedDict raises Error)
    """
    results = collections.OrderedDict()
    keys = outputs.keys()

    res_keys: t.List = [key for key in keys if 'res' in key]
    p_keys: t.List = sorted([key for key in keys if 'p' in key], reverse=True)

    new_keys = res_keys + p_keys

    for key in new_keys:
        results.update({key: outputs[key]})

    return results


#####################################################
def _create_model_out_dictkeys():
    """
    unsafe to use, will download all models;

    """
    model_names = []
    result_keys = []
    for model_name in model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys():
        try:
            print(model_name, ":")
            select_model(model_name)
            result = get_features_by_image_path("./sample.jpg")
            model_names.append(model_name)
            result_keys.append(list(result.keys()))
        except RuntimeError as t:
            print(t)

    pd.DataFrame(list(zip(model_names, result_keys))).to_csv("d2_model_out_dictkeys.csv")


