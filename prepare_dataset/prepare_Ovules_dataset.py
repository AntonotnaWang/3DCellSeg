import os
import sys

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
import matplotlib.pyplot as plt
import torch
from skimage import io
import h5py
import argparse

from func.dataset_preprocess import process_one_cuboid

parser = argparse.ArgumentParser(description='prepare_Ovules_dataset')
parser.add_argument('--source_file_path', default="/data0/wangad/CellSeg_dataset/Ovules", type=str,
                        help='')
parser.add_argument('--output_file_path', default="/data0/wangad/CellSeg_dataset/Ovules_processed_thin_boundary", type=str,
                        help='')
parser.add_argument('--img_size_scale_factor', default=0.5, type=float,
                        help='')
parser.add_argument('--width_of_membrane', default=1, type=float,
                        help='')
args = parser.parse_args()

source_file_path = args.source_file_path
output_file_path = args.output_file_path
scale_factor = args.img_size_scale_factor

# the following is to create dict = each item is {picname: {raw file path, seg file path}}

pic_dict = dict()

for category in ["train", "test", "val"]:
    pic_dict[category]={}
    case_names = os.listdir(source_file_path+'/'+category)
    for case_name in case_names:
        pic_dict[category][case_name] = source_file_path+'/'+category+'/'+case_name
        print('img '+str(source_file_path+'/'+category+'/'+case_name))

# the following is to create a pre-processed dataset for training

def process_one(raw_img, seg_img):
    raw_img=np.array(raw_img)
    seg_img=seg_img-np.min(seg_img)
    assert np.min(seg_img)==0
    background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = process_one_cuboid(seg_img, width_of_membrane = args.width_of_membrane)
    
    background_3d_mask = background_3d_mask.astype(np.uint8)
    foreground_3d_mask = foreground_3d_mask.astype(np.float32)
    boundary_3d_mask = boundary_3d_mask.astype(np.uint8)
    cell_ins_3d_mask = cell_ins_3d_mask.astype(np.uint16)
    
    return raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict

def img_3d_interpolate(img_3d, output_size, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1,1,img_3d.shape[0],img_3d.shape[1],img_3d.shape[2])
    img_3d=torch.from_numpy(img_3d).float().to(device)
    img_3d=torch.nn.functional.interpolate(img_3d, size=output_size, mode='nearest')
    img_3d=img_3d.detach().cpu().numpy()
    img_3d=img_3d.reshape(img_3d.shape[2],img_3d.shape[3],img_3d.shape[4])
    
    return img_3d

device = torch.device('cpu')

if not os.path.exists(output_file_path):
    os.mkdir(output_file_path)

for category in pic_dict.keys():
    if not os.path.exists(output_file_path+"/"+category):
        os.mkdir(output_file_path+"/"+category)
    for case_name in pic_dict[category].keys():
        print("processing: "+pic_dict[category][case_name])
        hf = h5py.File(pic_dict[category][case_name], 'r')
        #print(np.array(hf["segmentation"]))
        label = np.array(hf["label"], dtype=np.int)
        raw = np.array(hf["raw"], dtype=np.float32)
        hf.close()
        
        print("org raw size: "+str(raw.shape))
        print("org label size: "+str(label.shape))
        
        org_raw_img_shape = raw.shape
        output_size = (int(org_raw_img_shape[0]*scale_factor), int(org_raw_img_shape[1]*scale_factor), int(org_raw_img_shape[2]*scale_factor))
        raw = img_3d_interpolate(raw, output_size = output_size, device = device)
        label = img_3d_interpolate(label, output_size = output_size, device = device)
        
        raw = np.array(raw, dtype=np.float)
        label = np.array(label, dtype=np.int)
        
        print("raw size: "+str(raw.shape))
        print("label size: "+str(label.shape))
        
        raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = \
        process_one(raw, label)
        
#         print(raw_img.dtype)
#         print(background_3d_mask.dtype)
#         print(boundary_3d_mask.dtype)
#         print(foreground_3d_mask.dtype)
#         print(cell_ins_3d_mask.dtype)
        
        np.savez(output_file_path+"/"+category+"/"+case_name.split(".")[0]+".npz",
                            raw=raw_img,
                            ins=cell_ins_3d_mask,
                            boundary=boundary_3d_mask,
                            foreground=foreground_3d_mask)
        
        """
        h = h5py.File(output_file_path+"/"+category+"/"+case_name, 'w')
        h.create_dataset("raw",data=raw_img)
        h.create_dataset("ins",data=cell_ins_3d_mask)
        #h.create_dataset("background",data=background_3d_mask)
        h.create_dataset("boundary",data=boundary_3d_mask)
        h.create_dataset("foreground",data=foreground_3d_mask)
        """