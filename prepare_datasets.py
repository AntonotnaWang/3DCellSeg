import numpy as np
import os



##### HMS #####
source_path_raw_img = "/data/CellSeg_dataset/HMS/raw"
source_path_seg_img = "/data/CellSeg_dataset/HMS/segmentation_delete_fake_cells"
processed_path_raw_img = "/data/CellSeg_dataset/HMS_processed/raw"
processed_path_seg_img = "/data/CellSeg_dataset/HMS_processed/segmentation"

# step 1
run_py_script = "python prepare_dataset/prepare_HMS_dataset.py "+\
"--raw_img_file_path "+source_path_raw_img+" "+\
"--seg_img_file_path "+source_path_seg_img+" "+\
"--processed_raw_img_output_file_path "+processed_path_raw_img+" "+\
"--processed_seg_img_output_file_path "+processed_path_seg_img
print("run "+run_py_script)
os.system(run_py_script)

# step2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_HMS.py "+\
"--processed_raw_img_file_path "+processed_path_raw_img+" "+\
"--processed_seg_img_file_path "+processed_path_seg_img
print("run "+run_py_script)
os.system(run_py_script)
##### HMS #####



##### ATAS #####
# please change if needed
source_path = "/data/CellSeg_dataset/ATAS"
processed_path = "/data/CellSeg_dataset/ATAS_processed"
pre_cropped_path = "/data/CellSeg_dataset/ATAS_processed_pre_croped"

# step 1
run_py_script = "python prepare_dataset/prepare_ATAS_dataset.py "+\
"--source_file_path "+source_path+" "+\
"--output_file_path "+processed_path+" "+\
"--width_of_membrane 1.5"
print("run "+run_py_script)
os.system(run_py_script)

# step 2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_ATAS.py "+\
"--path "+processed_path+" "+\
"--test_name 'plant15' "
print("run "+run_py_script)
os.system(run_py_script)

# step 3
# use the dataset_info generated from get_the_dataset_info_of_ATAS.py
run_py_script = "python prepare_dataset/pre_crop_ATAS.py "+\
"--output_path "+pre_cropped_path+" "
print("run "+run_py_script)
os.system(run_py_script)

# step 4
run_py_script = "python prepare_dataset/get_the_dataset_info_of_ATAS_pre_cropped.py "+\
"--path "+pre_cropped_path
print("run "+run_py_script)
os.system(run_py_script)
##### ATAS #####



##### LRP #####
source_file_path = "/data/CellSeg_dataset/LateralRootPrimordia"
output_file_path = "/data/CellSeg_dataset/LateralRootPrimordia_processed_wide_boundary"
output_file_path_II = "/data/CellSeg_dataset/LateralRootPrimordia_pre_croped"

# step 1
run_py_script = "python prepare_dataset/prepare_LateralRootPrimordia_dataset.py "+\
"--source_file_path "+source_file_path+" "+\
"--output_file_path "+output_file_path+" "+\
"--img_size_scale_factor 0.5 "+\
"--width_of_membrane 2.5"
print("run "+run_py_script)
os.system(run_py_script)

# # or you can precrop the dataset
# # run_py_script = "python prepare_dataset/pre_crop_LateralRootPrimordia.py "+\
# # "--source_file_path "+source_file_path+" "+\
# # "--output_file_path "+output_file_path_II
# # print("run "+run_py_script)
# # os.system(run_py_script)

# step 2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_LateralRootPrimordia.py "+\
"--path "+output_file_path
print("run "+run_py_script)
os.system(run_py_script)
##### LRP #####



##### Ovules #####
source_file_path = "/data/CellSeg_dataset/Ovules"
output_file_path = "/data/CellSeg_dataset/Ovules_processed_thin_boundary"

# step 1
run_py_script = "python prepare_dataset/prepare_Ovules_dataset.py "+\
"--source_file_path "+source_file_path+" "+\
"--output_file_path "+output_file_path+" "+\
"--img_size_scale_factor 0.5 "+\
"--width_of_membrane 1"
print("run "+run_py_script)
os.system(run_py_script)

# step 2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_Ovules.py "+\
"--path "+output_file_path
print("run "+run_py_script)
os.system(run_py_script)
##### Ovules #####