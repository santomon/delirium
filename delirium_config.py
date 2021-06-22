"""
contains custom config for users and dataset
"""
import os

BOLD5K_ROI_DATA_PATH = os.path.join("/content", "ROIs")
BOLD5K_STIMULI_PATH = os.path.join("/content", "BOLD5000_Stimuli")

# src https://bold5000.github.io/download.html
BOLD5K_STIMULI_URL = "https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"
BOLD5K_ROI_DATA_URL = "https://ndownloader.figshare.com/files/12965447"


NN_DATA_PATH = os.path.join("/", "content", "output")

NN_RESULT_PATH = os.path.join("/", "content", "results")

NT_PATH = "./NeuralTaskonomy"

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



HEMISPHERES = ["LH", "RH"]
ROI = ["OPA", "PPA", "LOC", "EarlyVis", "RSC"]
ROI_LABELS = tuple(hs + roi for hs in HEMISPHERES for roi in ROI)

###############
taskonomy_features_urls = dict()
taskonomy_features_urls["rgb2sfnorm"] = "https://drive.google.com/u/0/uc?export=download&confirm=oakK&id=1WFb4F9a7BnqXbBKOP_xe77gDNiLlt4qZ"
taskonomy_features_urls["edge2d"] ="https://drive.google.com/u/0/uc?export=download&confirm=Oofu&id=18XrUM3UlwmX_RgYoJH1ycNvpLSb1JH0f"
taskonomy_features_urls["segmentsemantic"] = "https://drive.google.com/u/0/uc?export=download&confirm=WPRB&id=1KBqD2ccAMJi3hQhLHl2qBDuVsXoTzjRL"

################################
#downloadlinks:
# BOLD5000_Stimuli dataset:
# https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1

# BOLD5000 ROI data:
# https://ndownloader.figshare.com/files/12965447
