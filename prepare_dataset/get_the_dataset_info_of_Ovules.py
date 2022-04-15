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
parser = argparse.ArgumentParser(description='get_the_dataset_info_of_Ovules')
parser.add_argument('--path', default="/data0/wangad/CellSeg_dataset/Ovules_processed_small/", type=str,
                        help='')
args = parser.parse_args()
##### INPUT

Ovules_path = args.path
train_names = os.listdir(os.path.join(Ovules_path, "train"))
train_names = np.array(train_names)
val_names = os.listdir(os.path.join(Ovules_path, "val"))
val_names = np.array(val_names)
test_names = os.listdir(os.path.join(Ovules_path, "test"))
test_names = np.array(test_names)

Ovules_data_dict = {}
Ovules_data_dict["train"] = {}
Ovules_data_dict["test"] = {}

for name in train_names:
    Ovules_data_dict["train"][name] = os.path.join(Ovules_path, "train")+"/"+name
for name in val_names:
    Ovules_data_dict["train"][name] = os.path.join(Ovules_path, "val")+"/"+name
for name in test_names:
    Ovules_data_dict["test"][name] = os.path.join(Ovules_path, "test")+"/"+name

print(Ovules_data_dict)

save_obj(Ovules_data_dict, "dataset_info/Ovules_dataset_info")