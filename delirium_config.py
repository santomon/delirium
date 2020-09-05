"""
contains custom config for users and dataset
"""
import os

BOLD5K_ROI_DATA_PATH = r"/content/gdrive/My Drive/ROIs"
BOLD5K_STIMULI_PATH = r"/content/gdrive/My Drive/BOLD5000_Stimuli"

NN_DATA_PATH = r"C:\xd\bt\data\rcf_inference"
NN_SAVE_PATH = r"/content/output/deeplab_inference"

NT_PATH = r"C:\xd\bt\NeuralTaskonomy"

NAME_PREFIX = ""  # when loading inferences from neural nets, you can specify a prefix for the name of the file
NAME_SUFFIX = "_bb_f16"  # when loading inferences from neural nets, you can specify a prefix for the name of the file
NAME_FENDING = "npy"  # file-ending of the saving format
# name would be loaded as NAME_PREFIX + image_name + NAME_SUFFIX + NAME_FENDING

#################################
# dataset specific
BOLD5K_PRES_STIM_SUBPATH = os.path.join("Scene_Stimuli", "Presented_Stimuli")
BOLD5K_PRES_STIM_SUBDIRECTORIES = ("COCO", "ImageNet", "Scene")


################################
#downloadlinks:
# BOLD5000_Stimuli dataset:
# https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1

# BOLD5000 ROI data:
# https://ndownloader.figshare.com/files/12965447
