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
from medpy.io import load as medpyload
import matplotlib.pyplot as plt
import argparse

from func.dataset_preprocess import process_one_cuboid

parser = argparse.ArgumentParser(description='prepare_HMS_dataset')
parser.add_argument('--raw_img_file_path', default="/data/CellSeg_dataset/HMS/raw", type=str,
                        help='')
parser.add_argument('--seg_img_file_path', default="/data/CellSeg_dataset/HMS/segmentation_delete_fake_cells", type=str,
                        help='')
parser.add_argument('--processed_raw_img_output_file_path', default="/data/CellSeg_dataset/HMS_processed/raw", type=str,
                        help='')
parser.add_argument('--processed_seg_img_output_file_path', default="/data/CellSeg_dataset/HMS_processed/segmentation", type=str,
                        help='')
parser.add_argument('--width_of_membrane', default=1.5, type=float,
                        help='')
args = parser.parse_args()

raw_img_file_path=args.raw_img_file_path
seg_img_file_path=args.seg_img_file_path
process_raw_img_output_file_path=args.processed_raw_img_output_file_path
process_seg_img_output_file_path=args.processed_seg_img_output_file_path

raw_img_file_names=os.listdir(raw_img_file_path)
raw_img_file_names.sort()
seg_img_file_names=os.listdir(seg_img_file_path)
seg_img_file_names.sort()

# process seg_img
for file_name in seg_img_file_names:
    print('Processing seg img '+str(file_name))
    
    portion=os.path.splitext(file_name)
    file_name=portion[0]
    try:
        loading_file_name_path=seg_img_file_path+'/'+file_name+'.mha'
        img_3d, h=medpyload(loading_file_name_path)
    except:
        loading_file_name_path=seg_img_file_path+'/'+file_name+'.npy'
        img_3d = np.load(loading_file_name_path).astype(float)
    background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict=process_one_cuboid(img_3d, width_of_membrane = args.width_of_membrane)
    if not os.path.exists(process_seg_img_output_file_path+'/'+file_name):
        os.mkdir(process_seg_img_output_file_path+'/'+file_name)
    np.save(process_seg_img_output_file_path+'/'+file_name+'/'+file_name+'_background_3d_mask.npy', background_3d_mask)
    np.save(process_seg_img_output_file_path+'/'+file_name+'/'+file_name+'_boundary_3d_mask.npy', boundary_3d_mask)
    np.save(process_seg_img_output_file_path+'/'+file_name+'/'+file_name+'_foreground_3d_mask.npy', foreground_3d_mask)
    np.save(process_seg_img_output_file_path+'/'+file_name+'/'+file_name+'_ins.npy', cell_ins_3d_mask)


# process raw_img
for file_name in raw_img_file_names:
    print('Processing raw img '+str(file_name))
    
    portion=os.path.splitext(file_name)
    file_name=portion[0]
    loading_raw_file_name_path=raw_img_file_path+'/'+file_name+'.mha'
    raw_img_3d, seg_h=medpyload(loading_raw_file_name_path)
    try:
        loading_seg_file_name_path=seg_img_file_path+'/'+file_name+'.mha'
        seg_img_3d, seg_h=medpyload(loading_seg_file_name_path)
    except:
        loading_seg_file_name_path=seg_img_file_path+'/'+file_name+'.npy'
        seg_img_3d=np.load(loading_seg_file_name_path).astype(float)

    raw_img_3d[np.where(seg_img_3d==0)]=0
    np.save(process_raw_img_output_file_path+'/'+file_name+'.npy', raw_img_3d)