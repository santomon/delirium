import numpy as np
import os
import typing as t
import torch

from PIL import Image

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import detectron2.data.transforms as T

from detectron2.model_zoo import model_zoo


viable_models = model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()
# the viable model names can be found there
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py  # last checked 23.11.20



default_model = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"



device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg = get_cfg()


class FeatureExtractor:
    """

    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take RGB image as the input and apply conversion defined by `model.input_format`.
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

        self.cfg  = model_zoo.get_config(cfg_path)
        self.model = model_zoo.get(cfg_path)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
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
            if self.model.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            images = self.model.preprocess_image([inputs])

            predictions = self.model.backbone(images.tensor)
            if hasattr(self.model.backbone, "bottom_up"):
                predictions_bottom_up = self.model.backbone.bottom_up(images.tensor)
                predictions.update(predictions_bottom_up)
            return predictions


predictor: FeatureExtractor = FeatureExtractor(default_model)



def dependency_solver(): #implement this, if needed
    pass


def loader(path_to_image: str) -> np.ndarray:

    img = np.asarray(Image.open(path_to_image))  # loaded as RGB
    return img


def preprocessor(data_):
    pass


def model_call(data_):
    pass


def postprpocessor(data_):
    pass


def saver(data_, save_path, name):
    pass


def select_model(model_name: str):
    """
    change the model, that will be used by the predictor;
    to get the model_names, refer to viable models at the top of this file
    """
    global predictor
    predictor = FeatureExtractor(model_name)


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
        return predictor(image_data)


def _update_viable_models():
    model_names = []
    for model_name in model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys():
        try:
            print(model_name, ":")
            select_model(model_name)
            get_features_by_image_path("./input.jpg")
            model_names.append(model_name)
        except RuntimeError as t:
            print(t)
    return model_names
