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
from skimage import io
import argparse
import h5py

from func.dataset_preprocess import process_one_cuboid

parser = argparse.ArgumentParser(description='prepare_ATAS_dataset')
parser.add_argument('--source_file_path', default="/data0/wangad/CellSeg_dataset/ATAS", type=str,
                        help='')
parser.add_argument('--output_file_path', default="/data0/wangad/CellSeg_dataset/ATAS_processed", type=str,
                        help='')
parser.add_argument('--width_of_membrane', default=1.5, type=float,
                        help='')
args = parser.parse_args()

source_file_path = args.source_file_path
output_file_path = args.output_file_path

plant_files=os.listdir(source_file_path)
plant_files.sort()

pic_dict = dict()

for plant in plant_files:
    raw_img_file_path=source_file_path+'/'+plant+'/processed_tiffs'
    seg_img_file_path=source_file_path+'/'+plant+'/segmentation_tiffs'
    
    raw_img_file_names=os.listdir(raw_img_file_path)
    raw_img_file_names.sort()
    seg_img_file_names=os.listdir(seg_img_file_path)
    seg_img_file_names.sort()
        
    for file_name in raw_img_file_names:
        if 'acylYFP' in file_name:
            print('raw img '+str(plant+'_'+file_name.split('_')[0]))
            pic_dict[plant+'_'+file_name.split('_')[0]] = {"raw_img_path":raw_img_file_path+'/'+file_name}
            
            
    for file_name in seg_img_file_names:
        print('seg img '+str(plant+'_'+file_name.split('_')[0]))
        if plant+'_'+file_name.split('_')[0] in pic_dict.keys():
            pic_dict[plant+'_'+file_name.split('_')[0]]["seg_img_path"]=seg_img_file_path+'/'+file_name

def process_one(raw_img_path, seg_img_path):
    # raw img
    raw_img=io.imread(raw_img_path)
    raw_img=np.array(raw_img)
    
    #seg img
    seg_img=io.imread(seg_img_path)
    seg_img=seg_img-np.min(seg_img)
    assert np.min(seg_img)==0
    background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = process_one_cuboid(seg_img, width_of_membrane = args.width_of_membrane)
    
    background_3d_mask=np.array(background_3d_mask, dtype=np.uint8)
    boundary_3d_mask=np.array(boundary_3d_mask, dtype=np.uint8)
    cell_ins_3d_mask=np.array(cell_ins_3d_mask, dtype=np.uint16)
    
    return raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict

'''
raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = \
process_one(pic_dict['plant1_0hrs']['raw_img_path'], pic_dict['plant1_0hrs']['seg_img_path'])
'''

if not os.path.exists(output_file_path):
    os.mkdir(output_file_path)

for img_name in pic_dict.keys():
    
    if len(pic_dict[img_name])==2:

        print("processing: "+output_file_path+"/"+img_name)

        raw_img, background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict = \
        process_one(pic_dict[img_name]['raw_img_path'], pic_dict[img_name]['seg_img_path'])
        
        h = h5py.File(output_file_path+"/"+img_name+".h5", 'w')
        h.create_dataset("raw",data=raw_img)
        h.create_dataset("ins",data=cell_ins_3d_mask)
        h.create_dataset("background",data=background_3d_mask)
        h.create_dataset("boundary",data=boundary_3d_mask)
        h.create_dataset("foreground",data=foreground_3d_mask)
        
        """
        if not os.path.exists(output_file_path+"/"+img_name):
            os.mkdir(output_file_path+"/"+img_name)
        
        np.save(output_file_path+"/"+img_name+'/raw_img.npy', raw_img)
        np.save(output_file_path+"/"+img_name+'/ins_seg.npy', cell_ins_3d_mask)
        np.save(output_file_path+"/"+img_name+'/background_mask.npy', background_3d_mask)
        np.save(output_file_path+"/"+img_name+'/boundary_mask.npy', boundary_3d_mask)
        np.save(output_file_path+"/"+img_name+'/foreground_mask.npy', foreground_3d_mask)
        np.save(output_file_path+"/"+img_name+'/centers.npy', center_dict)
        """