# -*- coding: utf-8 -*-
"""

This is a script file that performs dataset preprocessing.

Mostly it prepares dataset structure for keras

"""
import os, sys
import random
import numpy as np
import glob


DATA_PATH = "data/PKLot/PKLotSegmented/"

all_images = [DATA_PATH+i for i in os.listdir(DATA_PATH)] # use this for full dataset

PARKING_LIST = ["PUC"]
CONDITION_LIST = ["Sunny"]

#PARKING_LIST = ["PUC", "UFPR04", "UFPR05"]
#CONDITION_LIST = ["Cloudy", "Rainy", "Sunny"]

def browse_dirs():
    dir_list = []
    for park in PARKING_LIST:
        for cond in CONDITION_LIST:
            dirs = glob.glob(os.path.join(DATA_PATH, park, cond, '*'))
            print("There are %d subdirectories in %s" % (len(dirs), os.path.join(DATA_PATH, park, cond)))
            dir_list.append(dirs)

    return dir_list

#browse_dirs()

SPLIT = np.array([0.5, 0.8])

def split_dirs(dir_list):
    split_list = []
    for dirs in np.array(dir_list):
        #print("Split indices %s" % ((len(dirs)*SPLIT).astype(int)))
        split_list.append(np.array_split(np.array(dirs), (len(dirs)*SPLIT).astype(int)))

    return split_list

#split_dirs(browse_dirs())

def create_dirs(dir_list):
    symlinks_list = ['data/symlinks/train', 'data/symlinks/validation', 'data/symlinks/test']
    if not os.path.exists('data/symlinks'):
        os.mkdir(d)
    for d in symlinks_list:
        if not os.path.exists(d):
            os.mkdir(d)
        if not os.path.exists(os.path.join(d, "Occupied")):
            os.mkdir(os.path.join(d, "Occupied"))
        if not os.path.exists(os.path.join(d, "Empty")):
            os.mkdir(os.path.join(d, "Empty"))

    for dirs in dir_list:
        for i, data in enumerate(dirs):
            for d in data:
                print(symlinks_list[i], len(d))
                occupied_list = glob.glob(os.path.join(d, "Occupied", "*.jpg"), recursive=True)
                empty_list = glob.glob(os.path.join(d, "Empty", "*.jpg"), recursive=True)
                print("Empty class in %s is: %d"%(d, len(empty_list)))
                print("Occupied class in %s is: %d"%(d, len(occupied_list)))
                create_symlinks(occupied_list, d, symlinks_list[i])
                create_symlinks(empty_list, d, symlinks_list[i])


def create_symlinks(file_list, root_path, sym_path):
    file = file_list
    for file in file_list:
        dst = file.replace(root_path, sym_path)
        src = file
        if not os.path.exists(os.path.abspath(dst)):
            os.symlink(os.path.abspath(src), os.path.abspath(dst))

create_dirs(split_dirs(browse_dirs()))









































