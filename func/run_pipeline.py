import numpy as np
import torch
from torchsummary import summary
from torch import from_numpy as from_numpy
from sklearn.cluster import DBSCAN
import edt
import copy
from skimage.segmentation import watershed

# reassign unique numbers
def reassign(seg_final):
    seg_final_unique_numbers=np.unique(seg_final)
    seg_final_unique_numbers=seg_final_unique_numbers[seg_final_unique_numbers>0]
    seg_final_unique_numbers_reshuffle=copy.deepcopy(seg_final_unique_numbers)
    np.random.shuffle(seg_final_unique_numbers_reshuffle)
    seg_final_copy=copy.deepcopy(seg_final)
    for i in np.arange(len(seg_final_unique_numbers)):
        print('reassign unique numbers progress: '+str(i/len(seg_final_unique_numbers)), end='\r')
        seg_final_copy[np.where(seg_final==seg_final_unique_numbers[i])]=seg_final_unique_numbers_reshuffle[i]*5
    
    return seg_final_copy

# assign_boudary_voxels_to_cells with watershed (faster)
def assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cells, seg_boundary, seg_background, compactness=1):
    mask = np.array(seg_background==0)
    marker = seg_foreground_single_cells
    
    """
    distance = 1 - seg_background - np.array(seg_foreground_single_cells>0, dtype=np.float)
    distance=np.array(distance, dtype=np.uint32, order='F')
    distance=edt.edt(distance,black_border=True, order='F',parallel=1)
    distance=np.min(distance)*np.array(seg_foreground_single_cells>0, dtype=np.float)+distance
    """
    distance_of_boundary=np.array(seg_boundary, dtype=np.uint32, order='F')
    distance_of_boundary=edt.edt(distance_of_boundary,black_border=True, order='F',parallel=1)

    distance_of_remaining_foreground = 1 - seg_background - np.array(seg_foreground_single_cells>0, dtype=np.float)
    distance_of_remaining_foreground=np.array(distance_of_remaining_foreground, dtype=np.uint32, order='F')
    distance_of_remaining_foreground=edt.edt(distance_of_remaining_foreground,black_border=True, order='F',parallel=1)

    distance = distance_of_remaining_foreground+np.max(distance_of_remaining_foreground)*distance_of_boundary
    distance=np.min(distance)*np.array(seg_foreground_single_cells>0, dtype=np.float)+distance
    
    labels = watershed(distance, marker, mask=mask, compactness=compactness)
    
    return labels

# assign_boudary_voxels_to_cells
def assign_boudary_voxels_to_cells(seg_foreground_single_cells, seg_background, device):
    cell_unique_values=np.unique(seg_foreground_single_cells)
    cell_unique_values=cell_unique_values[cell_unique_values>0]

    seg_foreground_final_dt=edt.edt(
        np.array(seg_foreground_single_cells>0, dtype=np.uint32, order='F'),
        black_border=True, order='F',
        parallel=1)

    boundary_locs=np.where((1-seg_background-(seg_foreground_single_cells>0))>0)
    boundary_loc_len=boundary_locs[0].shape[0]
    boundary_locs=np.array([boundary_locs[0],boundary_locs[1],boundary_locs[2]])

    for i, cell_unique_value in enumerate(cell_unique_values):
        temp_cell_locs=np.where(np.logical_and(seg_foreground_single_cells==cell_unique_value, seg_foreground_final_dt==1))
        temp_cell_loc_len=temp_cell_locs[0].shape[0]
        temp_cell_locs=np.array([temp_cell_locs[0],temp_cell_locs[1],temp_cell_locs[2]])

        if i==0:
            cell_locs=temp_cell_locs
            cell_tags=cell_unique_value*np.ones([1,temp_cell_loc_len])
        else:
            cell_locs=np.concatenate((cell_locs, temp_cell_locs), axis=1)
            cell_tags=np.concatenate((cell_tags, cell_unique_value*np.ones([1,temp_cell_loc_len])), axis=1)
    
    step=100
    cell_locs_shape=cell_locs.shape
    cell_locs_reshape=torch.tensor(cell_locs.reshape(1, cell_locs_shape[0], cell_locs_shape[1])).float().to(device)
    boundary_locs=torch.tensor(boundary_locs).float().to(device)
    cell_tags=torch.tensor(cell_tags).float().to(device)
    boundary_tag=torch.zeros(boundary_loc_len).to(device)

    for i_start in np.arange(0, boundary_loc_len, step):
        print('assign boudary voxels to their nearest cells progress: '+str(i_start/boundary_loc_len), end='\r')
        if i_start+step<boundary_loc_len:
            i_end=i_start+step
        else:
            i_end=boundary_loc_len
        temp_boundary_vos=boundary_locs[:,i_start:i_end].T
        temp_boundary_vos_shape=temp_boundary_vos.shape
        temp_boundary_vos=temp_boundary_vos.view(temp_boundary_vos_shape[0], temp_boundary_vos_shape[1], 1)
        distance=torch.sum((cell_locs_reshape-temp_boundary_vos)**2, dim=1)
        boundary_tag[i_start:i_end]=cell_tags[0, torch.argmin(distance, dim=1)]

    seg_boundary_with_tag=np.zeros(seg_foreground_single_cells.shape)
    boundary_locs=boundary_locs.cpu().detach().numpy()
    boundary_locs=np.array(boundary_locs, dtype=np.int)
    seg_boundary_with_tag[boundary_locs[0,:], boundary_locs[1,:], boundary_locs[2,:]]=boundary_tag.cpu().detach().numpy()
    
    seg_final=seg_boundary_with_tag+seg_foreground_single_cells
    
    return seg_final

# Stage 2: DBSCAN
def dbscan_of_seg(seg_cell, threshold=100, dbscan_min_samples=5, dbscan_eps=1):
    cell_locs=np.where(seg_cell>0)
    cell_locs_x=cell_locs[0]
    cell_locs_y=cell_locs[1]
    cell_locs_z=cell_locs[2]
    cell_locs_len=cell_locs[0].shape[0]
    cell_locs_reshape=np.concatenate((cell_locs[0].reshape(cell_locs_len,1),
                                      cell_locs[1].reshape(cell_locs_len,1),
                                      cell_locs[2].reshape(cell_locs_len,1)),axis=1)
    
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean').fit(cell_locs_reshape)
    clustering_labels=clustering.labels_
    clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
    
    clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>-1]
    clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>-1] # delete noise
    clustering_labels_unique=clustering_labels_unique+1
    
    clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
    clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
    seg_single_cells=np.zeros(seg_cell.shape)
    
    for i in range(0, len(clustering_labels_unique)):
        temp_label=clustering_labels_unique[i]
        temp_label_locs=np.where(clustering_labels==temp_label-1)
        seg_single_cells[cell_locs_x[temp_label_locs],cell_locs_y[temp_label_locs],cell_locs_z[temp_label_locs]]= \
        clustering_labels_unique[i]
    
    return seg_single_cells, clustering_labels_unique, clustering_labels_counts

def semantic_segment_crop_and_cat(raw_img, model, device, crop_cube_size=64, stride=64):
    # raw_img: 3d matrix, numpy.array
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
    
    for i in [0,1,2]:
        while crop_cube_size[i]>raw_img.shape[i]:
            crop_cube_size[i]=int(crop_cube_size[i]/2)
            stride[i]=crop_cube_size[i]
    
    img_shape=raw_img.shape
    
    seg_background=np.zeros(img_shape)
    seg_boundary=np.zeros(img_shape)
    seg_foreground=np.zeros(img_shape)
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    count=0
    
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('Progress of segment_3d_img: '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start_input=i
                    x_end_input=i+crop_cube_size[0]
                    x_start_output=i
                    x_end_output=i+stride[0]
                else:
                    x_start_input=img_shape[0]-crop_cube_size[0]
                    x_end_input=img_shape[0]
                    x_start_output=i
                    x_end_output=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start_input=j
                    y_end_input=j+crop_cube_size[1]
                    y_start_output=j
                    y_end_output=j+stride[1]
                else:
                    y_start_input=img_shape[1]-crop_cube_size[1]
                    y_end_input=img_shape[1]
                    y_start_output=j
                    y_end_output=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start_input=k
                    z_end_input=k+crop_cube_size[2]
                    z_start_output=k
                    z_end_output=k+stride[2]
                else:
                    z_start_input=img_shape[2]-crop_cube_size[2]
                    z_end_input=img_shape[2]
                    z_start_output=k
                    z_end_output=img_shape[2]
                
                raw_img_crop=raw_img[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                raw_img_crop=raw_img_crop.reshape(1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2])
                raw_img_crop=from_numpy(raw_img_crop).float().to(device)
                
                with torch.no_grad():
                    seg_crop_output=model(raw_img_crop)
                seg_crop_output_np=seg_crop_output.cpu().detach().numpy()
                
                seg_crop_output_np_bg=seg_crop_output_np[0,0,:,:,:]
                seg_crop_output_np_bd=seg_crop_output_np[0,1,:,:,:]
                seg_crop_output_np_fg=seg_crop_output_np[0,2,:,:,:]
                
                seg_background_temp=np.zeros(img_shape)
                seg_boundary_temp=np.zeros(img_shape)
                seg_foreground_temp=np.zeros(img_shape)
                
                seg_background_temp[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]=seg_crop_output_np_bg
                seg_boundary_temp[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]=seg_crop_output_np_bd
                seg_foreground_temp[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]=seg_crop_output_np_fg
                
                seg_background[x_start_output:x_end_output, y_start_output:y_end_output, z_start_output:z_end_output]=seg_background_temp[x_start_output:x_end_output, y_start_output:y_end_output, z_start_output:z_end_output]
                seg_boundary[x_start_output:x_end_output, y_start_output:y_end_output, z_start_output:z_end_output]=seg_boundary_temp[x_start_output:x_end_output, y_start_output:y_end_output, z_start_output:z_end_output]
                seg_foreground[x_start_output:x_end_output, y_start_output:y_end_output, z_start_output:z_end_output]=seg_foreground_temp[x_start_output:x_end_output, y_start_output:y_end_output, z_start_output:z_end_output]
                
                count=count+1
                
    return {'background': seg_background, 'boundary': seg_boundary, 'foreground': seg_foreground}

def segment(raw_img, model, device,
            crop_cube_size=256, stride=128,
            dbscan_min_samples=32, dbscan_eps=2, threshold=30):
    # process the raw img using model
    print('model', end='\r')
    raw_img_size=raw_img.shape
    with torch.no_grad():
        seg_img=\
        semantic_segment_crop_and_cat(raw_img, model, device, crop_cube_size=crop_cube_size, stride=stride)
    seg_img_background=seg_img['background']
    seg_img_boundary=seg_img['boundary']
    seg_img_foreground=seg_img['foreground']
    
    torch.cuda.empty_cache()
    
    # argmax
    print('argmax', end='\r')
    seg=[]
    seg.append(seg_img_background)
    seg.append(seg_img_boundary)
    seg.append(seg_img_foreground)
    seg=np.array(seg)
    seg_argmax=np.argmax(seg, axis=0)
    # probability map to 0 1 segment
    seg_background=np.zeros(raw_img.shape)
    seg_background[np.where(seg_argmax==0)]=1
    seg_foreground=np.zeros(raw_img.shape)
    seg_foreground[np.where(seg_argmax==2)]=1
    seg_boundary=np.zeros(raw_img.shape)
    seg_boundary[np.where(seg_argmax==1)]=1
    
    # DBSCAN processing
    print('DBSCAN', end='\r')
    seg_foreground_single_cells, clustering_labels_unique, clustering_labels_counts= \
    dbscan_of_seg(seg_cell=seg_foreground,
                  dbscan_min_samples=dbscan_min_samples,
                  dbscan_eps=dbscan_eps,
                  threshold=threshold)
    
    # assign boudary voxels to their nearest cells
    seg_final=assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cells, seg_boundary, seg_background, compactness=1)
    #seg_final=assign_boudary_voxels_to_cells(seg_foreground_single_cells, seg_background, device)
    #torch.cuda.empty_cache()
    
    # reassign unique numbers
    seg_final=reassign(seg_final)
    
    return seg_final