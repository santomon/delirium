import os
import typing as t

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
run_img_task_loc = "./taskonomy/taskbank/tools/run_img_task2.py"  # exptects cd to be delirium

command_dict: t.Dict = None

def select_model(model_name):
    global currently_selected_model, command_dict
    if model_name in viable_models:
        currently_selected_model = model_name
        command_dict = dict()
        command_dict['model_name'] = model_name
        command_dict['run_img_task_loc'] = run_img_task_loc
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
        command_dict['compress'] = '--compress-rep'
    else:
        command_dict['compress'] = ""

    return taskonomy_command_str


def saver(taskonomy_command_str, save_path: str, old_file_name: str):
    full_path = os.path.join(save_path, module_name, currently_selected_model)
    if not os.path.isdir(full_path):
        os.makedirs(full_path)
    command_dict['path_to_out'] = os.path.join(full_path, generate_file_name(old_file_name))
    os.system(taskonomy_command_str.format(**command_dict))


def generate_file_name(old_file_name):
    return old_file_name + "_features" + ".npy"


select_model(default_model)



