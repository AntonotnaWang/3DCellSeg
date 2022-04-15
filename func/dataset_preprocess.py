"""
3D background boundary foreground of one cuboid
"""
import numpy as np
import edt
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def revised_crop_and_stride(img_shape, crop_cube_size, stride):
    for i in range(len(img_shape)):
        if img_shape[i]<=crop_cube_size[i]:
            crop_cube_size[i]=img_shape[i]
            stride[i]=img_shape[i]
    return crop_cube_size, stride

def crop_one_3d_img(input_img, crop_cube_size, stride):
    # input_img: 3d matrix, numpy.array
    assert isinstance(crop_cube_size, (int, list))
    if isinstance(crop_cube_size, int):
        crop_cube_size=np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size)==3
    
    assert isinstance(stride, (int, list))
    if isinstance(stride, int):
        stride=np.array([stride, stride, stride])
    else:
        assert len(stride)==3
    
    img_shape=input_img.shape
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    
    count=0
    
    crop_list = []
    
    for i in np.arange(0, img_shape[0]-stride[0]+1, stride[0]):
        for j in np.arange(0, img_shape[1]-stride[1]+1, stride[1]):
            for k in np.arange(0, img_shape[2]-stride[2]+1, stride[2]):
                print('crop one 3d img progress : '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<img_shape[0]:
                    x_start_input=i
                    x_end_input=i+crop_cube_size[0]
                else:
                    x_start_input=img_shape[0]-crop_cube_size[0]
                    x_end_input=img_shape[0]
                
                if j+crop_cube_size[1]<img_shape[1]:
                    y_start_input=j
                    y_end_input=j+crop_cube_size[1]
                else:
                    y_start_input=img_shape[1]-crop_cube_size[1]
                    y_end_input=img_shape[1]
                
                if k+crop_cube_size[2]<img_shape[2]:
                    z_start_input=k
                    z_end_input=k+crop_cube_size[2]
                else:
                    z_start_input=img_shape[2]-crop_cube_size[2]
                    z_end_input=img_shape[2]
                
                if i-stride[0]+crop_cube_size[0]<img_shape[0] and \
                j-stride[1]+crop_cube_size[1]<img_shape[1] and k-stride[2]+crop_cube_size[2]<img_shape[2]:
                    #print("crop range: "+str((x_start_input, x_end_input, y_start_input, y_end_input, z_start_input, z_end_input)))
                    crop_temp=input_img[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                    crop_list.append(np.array(crop_temp))
                
                count=count+1
                
    return crop_list

def find_center_and_add_bounding_box(temp_cell, x_min, y_min, z_min):
    temp_cell_locs=np.where(temp_cell>0)
    if temp_cell_locs[0].size>0:
        temp_cell_x_max=np.max(temp_cell_locs[0])
        temp_cell_y_max=np.max(temp_cell_locs[1])
        temp_cell_z_max=np.max(temp_cell_locs[2])
        temp_cell_x_min=np.min(temp_cell_locs[0])
        temp_cell_y_min=np.min(temp_cell_locs[1])
        temp_cell_z_min=np.min(temp_cell_locs[2])

        temp_cell_x_len=(temp_cell_x_max-temp_cell_x_min)+x_min
        temp_cell_y_len=(temp_cell_y_max-temp_cell_y_min)+y_min
        temp_cell_z_len=(temp_cell_z_max-temp_cell_z_min)+z_min

        temp_cell_center_x=temp_cell_x_len/2
        temp_cell_center_y=temp_cell_y_len/2
        temp_cell_center_z=temp_cell_z_len/2

        return {'center_x': temp_cell_center_x,
                'center_y': temp_cell_center_y,
                'center_z': temp_cell_center_z,
                'x_len': temp_cell_x_len,
                'y_len': temp_cell_y_len,
                'z_len': temp_cell_z_len}
    else:
        return 0

def distance_trans(img_3d):
    img_3d[img_3d>0]=1
    img_3d=np.array(img_3d, dtype=np.uint32, order='F')
    img_3d_dt=edt.edt(
        img_3d,
        black_border=True, order='F',
        parallel=1)
    return img_3d_dt

def dbscan(img_3d, eps=1, min_samples=1, threshold=1):
    cell_locs=np.where(img_3d>0)
    cell_locs_x=cell_locs[0]
    cell_locs_y=cell_locs[1]
    cell_locs_z=cell_locs[2]
    cell_locs_len=cell_locs[0].shape[0]
    cell_locs_reshape=np.concatenate((cell_locs[0].reshape(cell_locs_len,1),
                                      cell_locs[1].reshape(cell_locs_len,1),
                                      cell_locs[2].reshape(cell_locs_len,1)),axis=1)
        
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(cell_locs_reshape)
    clustering_labels=clustering.labels_
    clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
        
    clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>-1]
    clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>-1] # delete noise
    clustering_labels_unique+=1
    
    clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
    clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
    seg_single_cells=np.zeros(img_3d.shape)
    
    for i in range(0, len(clustering_labels_unique)):
        temp_label=clustering_labels_unique[i]
        temp_label_locs=np.where(clustering_labels==temp_label-1)
        seg_single_cells[cell_locs_x[temp_label_locs],cell_locs_y[temp_label_locs],cell_locs_z[temp_label_locs]]= \
        temp_label
    
    return seg_single_cells, clustering_labels_unique, clustering_labels_counts

def process_one_cuboid(input_3d_image, width_of_membrane=1.5, need_cell_center_info = False):
    input_3d_image=np.array(input_3d_image)
    # background 3D mask
    #--------------------------------------------------
    background_3d_mask=np.zeros(input_3d_image.shape)
    background_3d_mask[input_3d_image==0]=1
    #--------------------------------------------------
    
    # boundary, foreground, and cell_instance mask
    #--------------------------------------------------
    input_3d_image_shape=input_3d_image.shape
    boundary_3d_mask=np.zeros(input_3d_image_shape)
    foreground_3d_mask=np.zeros(input_3d_image_shape)
    cell_ins_3d_mask=np.zeros(input_3d_image_shape)
    
    # center dict
    center_dict = dict()
    
    mask_unique_values, mask_unique_counts=np.unique(input_3d_image, return_counts=True)
    mask_unique_values=np.array(mask_unique_values)
    mask_unique_counts=np.array(mask_unique_counts)
    
    # 0 means background, no mask
    mask_unique_counts=mask_unique_counts[mask_unique_values>0]
    mask_unique_values=mask_unique_values[mask_unique_values>0]
    
    for i in mask_unique_values:
        locs=np.where(input_3d_image==i)
        x_max=np.max(locs[0])
        y_max=np.max(locs[1])
        z_max=np.max(locs[2])
        x_min=np.min(locs[0])
        y_min=np.min(locs[1])
        z_min=np.min(locs[2])
        
        print('progress: '+str(i/len(mask_unique_values)), end='\r')
        
        temp_3d_img=copy.deepcopy(input_3d_image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1])
        temp_3d_img[np.where(temp_3d_img!=i)]=0
        temp_3d_img_dt = distance_trans(temp_3d_img)

        k=width_of_membrane
        
        # boundary
        temp_boundary_3d_mask=copy.deepcopy(boundary_3d_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1])
        temp_boundary_3d_mask[np.where(np.logical_and(temp_3d_img_dt>=1, temp_3d_img_dt<=k))]=1
        #temp_boundary_3d_mask[np.where(temp_3d_img_dt==1)]=1
        boundary_3d_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]=temp_boundary_3d_mask
        
        # foreground
        temp_foreground_3d_mask=copy.deepcopy(foreground_3d_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1])
        temp_foreground_3d_mask[np.where(temp_3d_img_dt>k)]=1 #temp_3d_img_dt[np.where(temp_3d_img_dt>k)]        
        foreground_3d_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]=temp_foreground_3d_mask
        
        # cell instances
        seg_single_cells, clustering_labels_unique, clustering_labels_counts = \
        dbscan(temp_3d_img_dt)
        seg_single_cells[seg_single_cells>0] = seg_single_cells[seg_single_cells>0] + i*3
        clustering_labels_unique = clustering_labels_unique + i*3
        cell_ins_3d_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]+=seg_single_cells
        
        if need_cell_center_info:
            # centers
            for i in range(0, len(clustering_labels_unique)):
                temp_label = clustering_labels_unique[i]
                temp_f_3d_mask = np.zeros(foreground_3d_mask.shape)
                temp_f_3d_mask[np.where(cell_ins_3d_mask==temp_label)] = foreground_3d_mask[np.where(cell_ins_3d_mask==temp_label)]
                radius = np.max(temp_f_3d_mask)
                center_loc = np.where(temp_f_3d_mask==radius)
                del temp_f_3d_mask
                center_loc = (center_loc[0][0], center_loc[1][0], center_loc[2][0])
                center_dict[temp_label] = {"radius": radius, "center_loc": center_loc}
            
    seg_img_fg=np.zeros(foreground_3d_mask.shape)
    seg_img_fg[np.where(foreground_3d_mask>0)]=1
    seg_img_fg=np.array(seg_img_fg, dtype=np.uint32, order='F')
    seg_img_fg_dt=edt.edt(
        seg_img_fg,
        black_border=True, order='F',
        parallel=1)
    del seg_img_fg
    seg_img_max=np.max(seg_img_fg_dt)
    seg_img_fg_dt=-seg_img_fg_dt+seg_img_max+1
    seg_img_fg_dt[np.where(seg_img_fg_dt==seg_img_max+1)]=0
    seg_img_fg_dt=seg_img_fg_dt/np.max(seg_img_fg_dt[np.where(seg_img_fg_dt>0)])
    foreground_3d_mask = seg_img_fg_dt
    del seg_img_fg_dt
    
    return background_3d_mask, boundary_3d_mask, foreground_3d_mask, cell_ins_3d_mask, center_dict