import os

BOLD5KSUBDIRS = ["COCO", "ImageNet", "Scene"]
FENDINGS = ["jpg", "JPEG", "JPG"]

def infer_recursive():
  """
    infers all
  """
  pass



def infer_folder(
  img_dir,
  out_dir,
  fendings = FENDINGS,
  model_ = None,
  model_params = [],
  compressor = None,
  compress_params = [],
):
  """
    given a list of file endings; applies the model for all those files
    in a specified directory
  """
  pass