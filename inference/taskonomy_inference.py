import os
import typing as t

import gdown

viable_output_layers = ['encoder/conv1',
                        'encoder/block1/unit_1/bottleneck_v1/shortcut',
                        'encoder/block1/unit_1/bottleneck_v1/conv1',
                        'encoder/block1/unit_1/bottleneck_v1/conv2',
                        'encoder/block1/unit_1/bottleneck_v1/conv3',
                        'encoder/block1/unit_1/bottleneck_v1',
                        'encoder/block1/unit_2/bottleneck_v1/conv1',
                        'encoder/block1/unit_2/bottleneck_v1/conv2',
                        'encoder/block1/unit_2/bottleneck_v1/conv3',
                        'encoder/block1/unit_2/bottleneck_v1',
                        'encoder/block1/unit_3/bottleneck_v1/conv1',
                        'encoder/block1/unit_3/bottleneck_v1/conv2',
                        'encoder/block1/unit_3/bottleneck_v1/conv3',
                        'encoder/block1/unit_3/bottleneck_v1',
                        'encoder/block1',
                        'encoder/block2/unit_1/bottleneck_v1/shortcut',
                        'encoder/block2/unit_1/bottleneck_v1/conv1',
                        'encoder/block2/unit_1/bottleneck_v1/conv2',
                        'encoder/block2/unit_1/bottleneck_v1/conv3',
                        'encoder/block2/unit_1/bottleneck_v1',
                        'encoder/block2/unit_2/bottleneck_v1/conv1',
                        'encoder/block2/unit_2/bottleneck_v1/conv2',
                        'encoder/block2/unit_2/bottleneck_v1/conv3',
                        'encoder/block2/unit_2/bottleneck_v1',
                        'encoder/block2/unit_3/bottleneck_v1/conv1',
                        'encoder/block2/unit_3/bottleneck_v1/conv2',
                        'encoder/block2/unit_3/bottleneck_v1/conv3',
                        'encoder/block2/unit_3/bottleneck_v1',
                        'encoder/block2/unit_4/bottleneck_v1/conv1',
                        'encoder/block2/unit_4/bottleneck_v1/conv2',
                        'encoder/block2/unit_4/bottleneck_v1/conv3',
                        'encoder/block2/unit_4/bottleneck_v1',
                        'encoder/block2',
                        'encoder/block3/unit_1/bottleneck_v1/shortcut',
                        'encoder/block3/unit_1/bottleneck_v1/conv1',
                        'encoder/block3/unit_1/bottleneck_v1/conv2',
                        'encoder/block3/unit_1/bottleneck_v1/conv3',
                        'encoder/block3/unit_1/bottleneck_v1',
                        'encoder/block3/unit_2/bottleneck_v1/conv1',
                        'encoder/block3/unit_2/bottleneck_v1/conv2',
                        'encoder/block3/unit_2/bottleneck_v1/conv3',
                        'encoder/block3/unit_2/bottleneck_v1',
                        'encoder/block3/unit_3/bottleneck_v1/conv1',
                        'encoder/block3/unit_3/bottleneck_v1/conv2',
                        'encoder/block3/unit_3/bottleneck_v1/conv3',
                        'encoder/block3/unit_3/bottleneck_v1',
                        'encoder/block3/unit_4/bottleneck_v1/conv1',
                        'encoder/block3/unit_4/bottleneck_v1/conv2',
                        'encoder/block3/unit_4/bottleneck_v1/conv3',
                        'encoder/block3/unit_4/bottleneck_v1',
                        'encoder/block3/unit_5/bottleneck_v1/conv1',
                        'encoder/block3/unit_5/bottleneck_v1/conv2',
                        'encoder/block3/unit_5/bottleneck_v1/conv3',
                        'encoder/block3/unit_5/bottleneck_v1',
                        'encoder/block3/unit_6/bottleneck_v1/conv1',
                        'encoder/block3/unit_6/bottleneck_v1/conv2',
                        'encoder/block3/unit_6/bottleneck_v1/conv3',
                        'encoder/block3/unit_6/bottleneck_v1',
                        'encoder/block3',
                        'encoder/block4/unit_1/bottleneck_v1/shortcut',
                        'encoder/block4/unit_1/bottleneck_v1/conv1',
                        'encoder/block4/unit_1/bottleneck_v1/conv2',
                        'encoder/block4/unit_1/bottleneck_v1/conv3',
                        'encoder/block4/unit_1/bottleneck_v1',
                        'encoder/block4/unit_2/bottleneck_v1/conv1',
                        'encoder/block4/unit_2/bottleneck_v1/conv2',
                        'encoder/block4/unit_2/bottleneck_v1/conv3',
                        'encoder/block4/unit_2/bottleneck_v1',
                        'encoder/block4/unit_3/bottleneck_v1/conv1',
                        'encoder/block4/unit_3/bottleneck_v1/conv2',
                        'encoder/block4/unit_3/bottleneck_v1/conv3',
                        'encoder/block4/unit_3/bottleneck_v1',
                        'encoder/block4',
                        'encoder_output']

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization jigsaw \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
viable_models = list_of_tasks.split()

module_name: str = "taskonomy_inference"
default_model = "edge2d"
default_output_layer = "encoder_output"

currently_selected_model: str = None
path_to_taskonomy = "./taskonomy"
run_img_task_loc = os.path.join(path_to_taskonomy, "taskbank", "tools", "run_img_task2.py")  # exptects cd to be delirium

command_dict: t.Dict = None

def select_model(model_name):
    global currently_selected_model, command_dict
    if model_name in viable_models:
        currently_selected_model = model_name
        command_dict = dict()
        command_dict['model_name'] = model_name
        command_dict['run_img_task_loc'] = run_img_task_loc
        _download_model(model_name)
    else:
        raise NotImplementedError("a task of this name is not available; refer to list_of_tasks for a list of viable tasks")


def loader(path_to_image: str):
    taskonomy_command_str = "python {run_img_task_loc} --img {path_to_img} --store {path_to_out} --store-rep --task {model_name} --encoder_layer {layer} {compress}"
    command_dict['path_to_img'] = path_to_image
    return taskonomy_command_str


def preprocessor(taskonomy_command_str):
    return taskonomy_command_str


def model_call(taskonomy_command_str, layer: str):
    command_dict['layer'] = layer
    return taskonomy_command_str


def postprocessor(taskonomy_command_str, compress=True):
    if compress:
        command_dict['compress'] = '--compress_rep'
    else:
        command_dict['compress'] = ""

    return taskonomy_command_str


def saver(taskonomy_command_str, save_path: str, old_file_name: str):
    full_path = os.path.join(save_path, module_name, currently_selected_model)
    if not os.path.isdir(full_path):
        os.makedirs(full_path)
    command_dict['path_to_out'] = os.path.join(full_path, generate_file_name(old_file_name))
    os.system(taskonomy_command_str.format(**command_dict))


    os.system("cls" if os.name=="nt" else "clear")


def generate_file_name(old_file_name):
    return old_file_name.split(".")[0] + "_features" + ".png"  # saving is actually done by run_img_task2.py from taskonomy
                                                 # saving format for encoder output will be ".npy"
                                                 # image from result of network will be .png


def _download_model(model_name):

    f1 = os.path.join(path_to_taskonomy, "taskbank", "temp", model_name, "model.permanent-ckpt.data-00000-of-00001")
    f2 = os.path.join(path_to_taskonomy, "taskbank", "temp", model_name, "model.permanent-ckpt.meta")
    f3 = os.path.join(path_to_taskonomy, "taskbank", "temp", model_name, "model.permanent-ckpt.index")

    if not os.path.isdir(os.path.join(path_to_taskonomy, "taskbank", "temp", model_name)):
        os.makedirs(os.path.join(path_to_taskonomy, "taskbank", "temp", model_name))

    if not os.path.isfile(f1):
        gdown.download("http://downloads.cs.stanford.edu/downloads/taskonomy_taskbankv1_models/{}/model.permanent-ckpt.data-00000-of-00001".format(model_name),
              f1, quiet=False)
    if not os.path.isfile(f2):
        gdown.download("http://downloads.cs.stanford.edu/downloads/taskonomy_taskbankv1_models/{}/model.permanent-ckpt.meta".format(model_name),
              f2, quiet=False)
    if not os.path.isfile(f3):
        gdown.download("http://downloads.cs.stanford.edu/downloads/taskonomy_taskbankv1_models/{}/model.permanent-ckpt.index".format(model_name),
              f3, quiet=False)



# select_model(default_model)



