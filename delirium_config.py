"""
contains custom config for users and dataset
"""

BOLD5K_ROI_DATA_PATH = r"C:\xd\bt\data\ROIs"
BOLD5K_STIMULI_PATH = r"C:\xd\bt\data\BOLD5000_Stimuli\BOLD5000_Stimuli"

NN_DATA_PATH = r"C:\xd\bt\data\rcf_inference"

NT_PATH = r"C:\xd\bt\NeuralTaskonomy"

NAME_PREFIX = ""  # when loading inferences from neural nets, you can specify a prefix for the name of the file
NAME_SUFFIX = "_bb_f16"  # when loading inferences from neural nets, you can specify a prefix for the name of the file
NAME_FENDING = "npy"  # file-ending of the saving format

#################################
# dataset specific
BOLD5K_PRES_STIM_SUBPATH = r"\Scene_Stimuli\Presented_Stimuli"
BOLD5K_PRES_STIM_SUBSETS = ["COCO", "ImageNet", "Scene"]


################################
# testing:
