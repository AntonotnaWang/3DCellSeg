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

import os
import numpy as np
from func.ultis import save_obj, load_obj
import argparse

##### INPUT
parser = argparse.ArgumentParser(description='get_the_dataset_info_of_HMS')
parser.add_argument('--processed_raw_img_file_path',
                    default="/data/CellSeg_dataset/HMS_processed/raw",
                    type=str, help='')
parser.add_argument('--processed_seg_img_file_path',
                    default="/data/CellSeg_dataset/HMS_processed/segmentation",
                    type=str, help='')
parser.add_argument('--instance_seg_img_file_path',
                    default="/data/CellSeg_dataset/HMS/segmentation/segmentation_revised/",
                    type=str, help='')
# choose plant15 as testing, others are used for training
parser.add_argument('--test_names', default=['120', '135', '65', '90'], nargs='+', type=str,
                        help='')
args = parser.parse_args()
##### INPUT

HMS_path_raw = args.processed_raw_img_file_path
HMS_path_seg = args.processed_seg_img_file_path
test_names = args.test_names

names = os.listdir(HMS_path_seg)
names = np.array(names)

train_names = []
for name in names:
    if name not in test_names:
        train_names.append(name)
train_names=np.array(train_names)

print("test: "+str(test_names))
print("train: "+str(train_names))

HMS_data_dict = {}
HMS_data_dict["train"] = {}
HMS_data_dict["test"] = {}

for name in names:
    if str(name) in test_names:
        HMS_data_dict["test"][name] = {}
        HMS_data_dict["test"][name]["raw"] = os.path.join(HMS_path_raw, name+".npy")
        HMS_data_dict["test"][name]["background"] = os.path.join(HMS_path_seg, name+"/"+name+"_background_3d_mask.npy")
        HMS_data_dict["test"][name]["boundary"] = os.path.join(HMS_path_seg, name+"/"+name+"_boundary_3d_mask.npy")
        HMS_data_dict["test"][name]["foreground"] = os.path.join(HMS_path_seg, name+"/"+name+"_foreground_3d_mask.npy")
        HMS_data_dict["test"][name]["ins"] = os.path.join(HMS_path_seg, name+"/"+name+"_ins.npy")
    elif str(name) in train_names:
        HMS_data_dict["train"][name] = {}
        HMS_data_dict["train"][name]["raw"] = os.path.join(HMS_path_raw, name+".npy")
        HMS_data_dict["train"][name]["background"] = os.path.join(HMS_path_seg, name+"/"+name+"_background_3d_mask.npy")
        HMS_data_dict["train"][name]["boundary"] = os.path.join(HMS_path_seg, name+"/"+name+"_boundary_3d_mask.npy")
        HMS_data_dict["train"][name]["foreground"] = os.path.join(HMS_path_seg, name+"/"+name+"_foreground_3d_mask.npy")

print(HMS_data_dict)
        
save_obj(HMS_data_dict, "dataset_info/HMS_dataset_info")
