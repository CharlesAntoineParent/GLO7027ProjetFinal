"""Module that define data separation flows."""


## Import
import os

from shutil import copyfile


## Data separation flows definition
def separate_images_15first_flow(dataset):
    dataset_path = dataset['data_path']
    train_path = dataset['train_path']
    test_path = dataset['test_path']
    
    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1


## Utils function
def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)