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
import matplotlib.pyplot as plt
import torch
import h5py
from skimage import io
import argparse

from func.dataset_preprocess import process_one_cuboid

parser = argparse.ArgumentParser(description='prepare_LateralRootPrimordia_dataset')
parser.add_argument('--source_file_path', default="/data0/wangad/CellSeg_dataset/LateralRootPrimordia", type=str,
                        help='')
parser.add_argument('--output_file_path', default="/data0/wangad/CellSeg_dataset/LateralRootPrimordia_processed_wide_boundary", type=str,
                        help='')
parser.add_argument('--img_size_scale_factor', default=0.5, type=float,
                        help='')
parser.add_argument('--width_of_membrane', default=2.5, type=float,
                        help='')

args = parser.parse_args()

source_file_path = args.source_file_path
output_file_path = args.output_file_path
scale_factor = args.img_size_scale_factor

# the following is to create dict = each item is {picname: {raw file path, seg file path}}

pic_dict = dict()

for category in ["train", "test", "val", "nuclei"]:
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
    
    background_3d_mask=np.array(background_3d_mask, dtype=np.uint8)
    boundary_3d_mask=np.array(boundary_3d_mask, dtype=np.uint8)
    cell_ins_3d_mask=np.array(cell_ins_3d_mask, dtype=np.uint16)
    
    return raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict

def img_3d_interpolate(img_3d, output_size, device = torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1,1,img_3d.shape[0],img_3d.shape[1],img_3d.shape[2])
    img_3d=torch.from_numpy(img_3d).float().to(device)
    img_3d=torch.nn.functional.interpolate(img_3d, size=output_size, mode='nearest')
    img_3d=img_3d.detach().cpu().numpy()
    img_3d=img_3d.reshape(img_3d.shape[2],img_3d.shape[3],img_3d.shape[4])
    
    return img_3d

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
        raw = np.array(hf["raw"], dtype=np.float)
        hf.close()
        
        print("org raw size: "+str(raw.shape))
        print("org label size: "+str(label.shape))
        
        org_raw_img_shape = raw.shape
        output_size = (int(org_raw_img_shape[0]*scale_factor), int(org_raw_img_shape[1]*scale_factor), int(org_raw_img_shape[2]*scale_factor))
        raw = img_3d_interpolate(raw, output_size = output_size)
        label = img_3d_interpolate(label, output_size = output_size)
        
        raw = np.array(raw, dtype=np.float)
        label = np.array(label, dtype=np.int)
        
        print("raw size: "+str(raw.shape))
        print("label size: "+str(label.shape))
        raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = \
        process_one(raw, label)
        
        h = h5py.File(output_file_path+"/"+category+"/"+case_name, 'w')
        h.create_dataset("raw",data=raw_img)
        h.create_dataset("ins",data=cell_ins_3d_mask)
        h.create_dataset("background",data=background_3d_mask)
        h.create_dataset("boundary",data=boundary_3d_mask)
        h.create_dataset("foreground",data=foreground_3d_mask)
        h.close()
        """
        if not os.path.exists(output_file_path+"/"+category+"/"+case_name):
            os.mkdir(output_file_path+"/"+category+"/"+case_name)
        io.imsave(output_file_path+"/"+category+"/"+case_name+'/raw_img.tif', raw_img)
        io.imsave(output_file_path+"/"+category+"/"+case_name+'/ins_seg.tif', cell_ins_3d_mask)
        io.imsave(output_file_path+"/"+category+"/"+case_name+'/background_mask.tif', background_3d_mask)
        io.imsave(output_file_path+"/"+category+"/"+case_name+'/boundary_mask.tif', boundary_3d_mask)
        io.imsave(output_file_path+"/"+category+"/"+case_name+'/foreground_mask.tif', foreground_3d_mask)
        np.save(output_file_path+"/"+category+"/"+case_name+'/centers.npy', center_dict)
        """