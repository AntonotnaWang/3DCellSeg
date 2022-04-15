import sys
import os
  
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

print(sys.path)

import numpy as np
import argparse

from func.ultis import save_obj, load_obj

##### INPUT
parser = argparse.ArgumentParser(description='get_the_dataset_info_of_ATAS')
parser.add_argument('--path', default="/data0/wangad/CellSeg_dataset/ATAS_processed/", type=str,
                        help='')
# choose plant15 as testing, others are used for training
parser.add_argument('--test_name', default="plant15", type=str,
                        help='')
args = parser.parse_args()
##### INPUT

ATAS_path = args.path
test_name = args.test_name

# get the data dict of ATAS
names = os.listdir(ATAS_path)

ATAS_data_dict = {}
ATAS_data_dict["train"] = {}
ATAS_data_dict["test"] = {}

for name in names:
    if name.split("_")[0]==test_name:
        ATAS_data_dict["test"][name] = os.path.join(ATAS_path, name)
    else:
        ATAS_data_dict["train"][name] = os.path.join(ATAS_path, name)

print(ATAS_data_dict)
        
save_obj(ATAS_data_dict, "dataset_info/ATAS_dataset_info")