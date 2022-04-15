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
import gc
import h5py
import argparse

from func.ultis import save_obj, load_obj
from func.dataset_preprocess import revised_crop_and_stride, crop_one_3d_img

##### INPUT
parser = argparse.ArgumentParser(description='pre_crop_LateralRootPrimordia_dataset')
parser.add_argument('--source_file_path', default="/data0/wangad/CellSeg_dataset/LateralRootPrimordia", type=str,
                        help='')
parser.add_argument('--output_file_path', default="/data0/wangad/CellSeg_dataset/LateralRootPrimordia_pre_croped", type=str,
                        help='')
args = parser.parse_args()

# may change the crop size and stride here
crop_cube_size=[256, 256, 256]
stride=[128, 128, 128]
##### INPUT

source_file_path = args.source_file_path
output_file_path = args.output_file_path

if not os.path.exists(output_file_path):
    os.mkdir(output_file_path)

# the following is to create dict = each item is {picname: {raw file path, seg file path}}

pic_dict = dict()

for category in ["train", "test", "val", "nuclei"]:
    pic_dict[category]={}
    case_names = os.listdir(source_file_path+'/'+category)
    for case_name in case_names:
        pic_dict[category][case_name] = source_file_path+'/'+category+'/'+case_name
        print('img '+str(source_file_path+'/'+category+'/'+case_name))


def process_one(raw_img, seg_img, crop_cube_size=crop_cube_size, stride=stride):
    raw_img=np.array(raw_img)
    seg_img=seg_img-np.min(seg_img)
    assert np.min(seg_img)==0
    
    crop_cube_size, stride = revised_crop_and_stride(raw_img.shape, crop_cube_size, stride)
    print("image shape: "+str(raw_img.shape)+"; crop: "+str(crop_cube_size)+"; stride: "+str(stride))
    
    #print("crop raw img")
    raw_img_crop_list = crop_one_3d_img(raw_img, crop_cube_size=crop_cube_size, stride=stride)
    #print("crop seg_img")
    seg_img_crop_list = crop_one_3d_img(seg_img, crop_cube_size=crop_cube_size, stride=stride)
    
    background_3d_mask_crop_list = []
    boundary_3d_mask_crop_list = []
    foreground_3d_mask_crop_list = []
    cell_ins_3d_mask_crop_list = []
    center_dict_crop_list = []
    for idx, seg_img_crop in enumerate(seg_img_crop_list):
        print("crop "+str(idx), end="\r")
        background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = process_one_cuboid(seg_img_crop)
        background_3d_mask_crop_list.append(background_3d_mask)
        boundary_3d_mask_crop_list.append(boundary_3d_mask)
        foreground_3d_mask_crop_list.append(foreground_3d_mask)
        cell_ins_3d_mask_crop_list.append(cell_ins_3d_mask)
        center_dict_crop_list.append(center_dict)
    
    return raw_img_crop_list, background_3d_mask_crop_list, boundary_3d_mask_crop_list, foreground_3d_mask_crop_list, cell_ins_3d_mask_crop_list, center_dict_crop_list

for category in ["train"]:
    if not os.path.exists(output_file_path+"/"+category):
        os.mkdir(output_file_path+"/"+category)
    for case_name in pic_dict[category].keys():
        print("processing: "+pic_dict[category][case_name])
        hf = h5py.File(pic_dict[category][case_name], 'r')
        #print(np.array(hf["segmentation"]))
        label = np.array(hf["label"])
        raw = np.array(hf["raw"])
        hf.close()
        print("size: "+str(raw.shape))
        raw_img_crop_list, background_3d_mask_crop_list, boundary_3d_mask_crop_list, foreground_3d_mask_crop_list, \
        cell_ins_3d_mask_crop_list, center_dict_crop_list = process_one(raw, label, crop_cube_size=[256, 256, 256], stride=[128, 128, 128])
        
        for idx in range(len(raw_img_crop_list)):
            if not os.path.exists(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)):
                os.mkdir(output_file_path+"/"+category+"/"+case_name+"_"+str(idx))
        
            np.save(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)+'/raw_img.npy', raw_img_crop_list[idx])
            np.save(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)+'/ins_seg.npy', cell_ins_3d_mask_crop_list[idx])
            np.save(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)+'/background_mask.npy', background_3d_mask_crop_list[idx])
            np.save(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)+'/boundary_mask.npy', boundary_3d_mask_crop_list[idx])
            np.save(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)+'/foreground_mask.npy', foreground_3d_mask_crop_list[idx])
            np.save(output_file_path+"/"+category+"/"+case_name+"_"+str(idx)+'/centers.npy', center_dict_crop_list[idx])