"""
Useful codes for easy use
"""
import os
import pickle

def read_pkl(file_path):
    """
    read pickle file
    """
    with open(file_path, "rb") as fin:
        pkl_file = pickle.load(fin, encoding="latin1")  # for CIFAR-10, we need to use encoding to convert from py2 to py3
    return pkl_file    

def new_dir(folder_path):
    """
    construct a new folder
    """
    if os.path.exists(folder_path):
        print(f"{folder_path} has already existed!")
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")    