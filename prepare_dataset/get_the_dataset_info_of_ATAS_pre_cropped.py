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
parser.add_argument('--path', default="/data0/wangad/CellSeg_dataset/ATAS_processed_pre_croped", type=str,
                        help='')
args = parser.parse_args()
##### INPUT

ATAS_data_dict=load_obj("dataset_info/ATAS_dataset_info")

Precrop_dataset_path = args.path

name_list = os.listdir(Precrop_dataset_path)

data_dict = dict()

data_dict["train"]={}
data_dict["test"]={}

for name in name_list:
    print("process ", name)
    if name.split("_")[0]+"_"+name.split("_")[1] in list(ATAS_data_dict["train"].keys()):
        data_dict["train"][name]={}
        data_dict["train"][name]["raw"]=os.path.join(Precrop_dataset_path, name)+"/raw_img.npy"
        data_dict["train"][name]["foreground"]=os.path.join(Precrop_dataset_path, name)+"/foreground_mask.npy"
        data_dict["train"][name]["boundary"]=os.path.join(Precrop_dataset_path, name)+"/boundary_mask.npy"
    elif name.split("_")[0]+"_"+name.split("_")[1] in list(ATAS_data_dict["test"].keys()):
        data_dict["test"][name]={}
        data_dict["test"][name]["raw"]=os.path.join(Precrop_dataset_path, name)+"/raw_img.npy"
        data_dict["test"][name]["foreground"]=os.path.join(Precrop_dataset_path, name)+"/foreground_mask.npy"
        data_dict["test"][name]["boundary"]=os.path.join(Precrop_dataset_path, name)+"/boundary_mask.npy"

print(data_dict)
        
save_obj(data_dict, "dataset_info/ATAS_pre_cropped_dataset_info")