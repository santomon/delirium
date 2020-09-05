"""
contains custom config for users and dataset
"""
import os

BOLD5K_ROI_DATA_PATH = r"C:\xd\bt\data\ROIs"
BOLD5K_STIMULI_PATH = r"C:\xd\bt\data\BOLD5000_Stimuli\BOLD5000_Stimuli"

NN_DATA_PATH = r"C:\xd\bt\data\rcf_inference"
NN_SAVE_PATH = r"C:\xd\bt\output\deeplab_inference"

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
# testing:
