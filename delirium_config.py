"""
contains custom config for users and dataset
"""
import os

BOLD5K_ROI_DATA_PATH = os.path.join("/content", "ROIs")
BOLD5K_STIMULI_PATH = os.path.join("/content", "BOLD5000_Stimuli")

# src https://bold5000.github.io/download.html
BOLD5K_STIMULI_URL = "https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"
BOLD5K_ROI_DATA_URL = "https://ndownloader.figshare.com/files/12965447"


NN_DATA_PATH = os.path.join("/content/gdrive/My Drive/output/rcf_inference")
NN_SAVE_PATH = os.path.join("/content/output/rcf_inference")

NT_PATH = r"C:\xd\bt\NeuralTaskonomy"

NAME_PREFIX = ""  # when loading inferences from neural nets, you can specify a prefix for the name of the file
NAME_SUFFIX = "_bb_f16"  # when loading inferences from neural nets, you can specify a prefix for the name of the file
NAME_FENDING = "npy"  # file-ending of the saving format
# name would be loaded as NAME_PREFIX + image_name + NAME_SUFFIX + NAME_FENDING

#################################
# dataset specific
BOLD5K_PRES_STIM_SUBPATH = os.path.join("Scene_Stimuli", "Presented_Stimuli")
BOLD5K_PRES_STIM_SUBDIRECTORIES = ("COCO", "ImageNet", "Scene")

UNWANTED_IMAGES = ("golfcourse7.jpg", "childsroom7.jpg", "COCO_train2014_000000000625.jpg")  # due to inconsistent sizes
UNWANTED_SUBSTRS = ("rep_",)  # due to repetition; but our network models of interest are not capable of such feats


################################
#downloadlinks:
# BOLD5000_Stimuli dataset:
# https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1

# BOLD5000 ROI data:
# https://ndownloader.figshare.com/files/12965447
