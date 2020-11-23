import numpy as np
import os
import typing as t
import torch

from PIL import Image

from detectron2.detectron2.modeling import build_model
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.layers import ShapeSpec
from detectron2.detectron2.data import MetadataCatalog
from detectron2.detectron2.checkpoint import DetectionCheckpointer
from detectron2.detectron2.config import get_cfg

from detectron2.detectron2.model_zoo import model_zoo


available_models = model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()
# the viable model names can be found there

device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg = get_cfg()


class FeatureGetter:
    """
    --- straight up copy pasta from detectron2 code of DefaultPredictor,
    modified, so that the backbone output can be safely infered---

    call(input_img : opened by cv2.imread(), pyramid: str) --> 4-dim tensor representing the output of the selected pyramid;
    valid values for pyramid:'p2' 'p3' 'p4' 'p5' 'p6'


    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
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

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        self.default_input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, pyramid: str):
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
            if self.input_format == "RGB":
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


def dependency_solver(): #implement this, if needed
    pass


def loader(path_to_image: str) -> np.ndarray:

    img = np.asarray(Image.open(path_to_image))  # loaded as RGB
    return img[:,:,::-1]    # reverse color order; essentially RGB -> BGR, which is required for detectron2 models
                            # what's funny is, is that the preprocessing step of detectron2 converts it back to RGB...


def preprocessor(data_):
    pass


def model_call(data_):
    pass


def postprpocessor(data_):
    pass


def saver(data_, save_path, name):
    pass
