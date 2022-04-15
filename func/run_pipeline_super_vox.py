import numpy as np
import torch
from torchsummary import summary
from torch import from_numpy as from_numpy
import edt
import copy
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.feature import peak_local_max

def segment_super_vox_2_channel(raw_img, model, device,
            crop_cube_size=128, stride=64,
            how_close_are_the_super_vox_to_boundary=2,
            min_touching_area=30, min_touching_percentage=0.51,
            min_cell_size_threshold=100,
            transposes = [[0,1,2],[2,0,1],[0,2,1],[1,0,2]], reverse_transposes = [[0,1,2],[1,2,0],[0,2,1],[1,0,2]]):
    # feed the raw img to the model
    print('Feed raw img to model. Use different transposes')
    raw_img_size=raw_img.shape
    
    seg_boundary_comp = np.zeros(raw_img_size)
    
    for idx, transpose in enumerate(transposes):
        print(str(idx+1)+": Transpose the image to be: "+str(transpose))
        with torch.no_grad():
            seg_img=\
            semantic_segment_crop_and_cat_2_channel_output(raw_img.transpose(transpose), model, device, crop_cube_size=crop_cube_size, stride=stride)
        seg_img_boundary=seg_img['boundary']
        seg_img_foreground=seg_img['foreground']
        torch.cuda.empty_cache()
    
        # argmax
        print('argmax', end='\r')
        # probability map to 0 1 segment
        seg_foreground=np.array(seg_img_foreground-seg_img_boundary>0, dtype=np.int)
        seg_boundary=1 - seg_foreground
        
        seg_foreground=seg_foreground.transpose(reverse_transposes[idx])
        seg_boundary=seg_boundary.transpose(reverse_transposes[idx])
        
        seg_boundary_comp+=seg_boundary
        
    print("Get model semantic seg by combination")
    seg_boundary_comp = np.array(seg_boundary_comp>0, dtype=np.int)
    seg_foreground_comp = 1 - seg_boundary_comp
    
    # Generate super vox by watershed
    seg_foreground_erosion=1-img_3d_erosion_or_expansion(1-seg_foreground_comp, kernel_size=how_close_are_the_super_vox_to_boundary+1, device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion, connectivity=min_touching_area)
    
    # Super voxel clustering
    cluster_super_vox=Cluster_Super_Vox(min_touching_area=min_touching_area, min_touching_percentage=min_touching_percentage)
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img
    
    # Delete too small cells
    seg_foreground_single_cell_with_boundary = delete_too_small_cluster(seg_foreground_single_cell_with_boundary, threshold=min_cell_size_threshold)
    
    # Assign boudary voxels to their nearest cells
    seg_final=assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary, seg_boundary_comp, compactness=1)
    
    # Reassign unique numbers
    # seg_final=reassign(seg_final)
    
    return seg_final

def segment_super_vox_3_channel(raw_img, model, device,
            crop_cube_size=128, stride=64,
            how_close_are_the_super_vox_to_boundary=2,
            min_touching_area=30, min_touching_percentage=0.51,
            min_cell_size_threshold=10,
            transposes = [[0,1,2],[2,0,1],[0,2,1],[1,0,2]], reverse_transposes = [[0,1,2],[1,2,0],[0,2,1],[1,0,2]]):
    # feed the raw img to the model
    print('Feed raw img to model. Use different transposes')
    raw_img_size=raw_img.shape
    
    seg_background_comp = np.zeros(raw_img_size)
    seg_boundary_comp = np.zeros(raw_img_size)
    
    for idx, transpose in enumerate(transposes):
        print(str(idx+1)+": Transpose the image to be: "+str(transpose))
        with torch.no_grad():
            seg_img=\
            semantic_segment_crop_and_cat_3_channel_output(raw_img.transpose(transpose), model, device, crop_cube_size=crop_cube_size, stride=stride)
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
        seg_background=np.zeros(seg_img_background.shape)
        seg_background[np.where(seg_argmax==0)]=1
        seg_foreground=np.zeros(seg_img_foreground.shape)
        seg_foreground[np.where(seg_argmax==2)]=1
        seg_boundary=np.zeros(seg_img_boundary.shape)
        seg_boundary[np.where(seg_argmax==1)]=1
        
        seg_background=seg_background.transpose(reverse_transposes[idx])
        seg_foreground=seg_foreground.transpose(reverse_transposes[idx])
        seg_boundary=seg_boundary.transpose(reverse_transposes[idx])
        
        seg_background_comp+=seg_background
        seg_boundary_comp+=seg_boundary
    print("Get model semantic seg by combination")
    seg_background_comp = np.array(seg_background_comp>0, dtype=np.int)
    seg_boundary_comp = np.array(seg_boundary_comp>0, dtype=np.int)
    seg_foreground_comp = np.array(1 - seg_background_comp - seg_boundary_comp>0, dtype=np.int)
    
    # Generate super vox by watershed
    seg_foreground_erosion=1-img_3d_erosion_or_expansion(1-seg_foreground_comp, kernel_size=how_close_are_the_super_vox_to_boundary+1, device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion, connectivity=min_touching_area)
    
    # Super voxel clustering
    cluster_super_vox=Cluster_Super_Vox(min_touching_area=min_touching_area, min_touching_percentage=min_touching_percentage)
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img
    
    # Delete too small cells
    seg_foreground_single_cell_with_boundary = delete_too_small_cluster(seg_foreground_single_cell_with_boundary, threshold=min_cell_size_threshold)
    
    # Assign boudary voxels to their nearest cells
    seg_final=assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary, seg_boundary_comp, seg_background_comp, compactness=1)
    
    # Reassign unique numbers
    #seg_final=reassign(seg_final)
    
    return seg_final

def semantic_segment_crop_and_cat_2_channel_output(raw_img, model, device, crop_cube_size=64, stride=64):
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
    
    #seg_background=np.zeros(img_shape)
    seg_boundary=np.zeros(img_shape)
    seg_foreground=np.zeros(img_shape)
    seg_log=np.zeros(img_shape) # 0 means this pixel has not been segmented, 1 means this pixel has been
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    count=0
    
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('Progress of segment_3d_img: '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start=i
                    x_end=i+crop_cube_size[0]
                else:
                    x_start=img_shape[0]-crop_cube_size[0]
                    x_end=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start=j
                    y_end=j+crop_cube_size[1]
                else:
                    y_start=img_shape[1]-crop_cube_size[1]
                    y_end=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start=k
                    z_end=k+crop_cube_size[2]
                else:
                    z_start=img_shape[2]-crop_cube_size[2]
                    z_end=img_shape[2]
                
                raw_img_crop=raw_img[x_start:x_end, y_start:y_end, z_start:z_end]
                raw_img_crop=raw_img_crop.reshape(1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2])
                raw_img_crop=from_numpy(raw_img_crop).float().to(device)
                
                seg_log_crop=seg_log[x_start:x_end, y_start:y_end, z_start:z_end]
                #seg_background_crop=seg_background[x_start:x_end, y_start:y_end, z_start:z_end]
                seg_boundary_crop=seg_boundary[x_start:x_end, y_start:y_end, z_start:z_end]
                seg_foreground_crop=seg_foreground[x_start:x_end, y_start:y_end, z_start:z_end]
                
                with torch.no_grad():
                    seg_crop_output=model(raw_img_crop)
                seg_crop_output_np=seg_crop_output.cpu().detach().numpy()
                
                #seg_crop_output_np_bg=seg_crop_output_np[0,0,:,:,:]
                seg_crop_output_np_bd=seg_crop_output_np[0,1,:,:,:]
                seg_crop_output_np_fg=seg_crop_output_np[0,0,:,:,:]
                
                #seg_background_temp=np.zeros(seg_background_crop.shape)
                seg_boundary_temp=np.zeros(seg_boundary_crop.shape)
                seg_foreground_temp=np.zeros(seg_foreground_crop.shape)
                
                #seg_background_temp[seg_log_crop==1]=(seg_crop_output_np_bg[seg_log_crop==1]+seg_background_crop[seg_log_crop==1])
                #seg_background_temp[seg_log_crop==0]=seg_crop_output_np_bg[seg_log_crop==0]
                
                seg_boundary_temp[seg_log_crop==1]=(seg_crop_output_np_bd[seg_log_crop==1]+seg_boundary_crop[seg_log_crop==1])/2
                seg_boundary_temp[seg_log_crop==0]=seg_crop_output_np_bd[seg_log_crop==0]
                
                seg_foreground_temp[seg_log_crop==1]=(seg_crop_output_np_fg[seg_log_crop==1]+seg_foreground_crop[seg_log_crop==1])/2
                seg_foreground_temp[seg_log_crop==0]=seg_crop_output_np_fg[seg_log_crop==0]
                
                #seg_background[x_start:x_end, y_start:y_end, z_start:z_end]=seg_background_temp
                seg_boundary[x_start:x_end, y_start:y_end, z_start:z_end]=seg_boundary_temp
                seg_foreground[x_start:x_end, y_start:y_end, z_start:z_end]=seg_foreground_temp
                
                seg_log[x_start:x_end, y_start:y_end, z_start:z_end]=1
                
                count=count+1
                
    return {'boundary': seg_boundary, 'foreground': seg_foreground}#{'background': seg_background, 'boundary': seg_boundary, 'foreground': seg_foreground}

def semantic_segment_crop_and_cat_3_channel_output(raw_img, model, device, crop_cube_size=64, stride=64):
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
    seg_log=np.zeros(img_shape) # 0 means this pixel has not been segmented, 1 means this pixel has been
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    count=0
    
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('Progress of segment_3d_img: '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start=i
                    x_end=i+crop_cube_size[0]
                else:
                    x_start=img_shape[0]-crop_cube_size[0]
                    x_end=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start=j
                    y_end=j+crop_cube_size[1]
                else:
                    y_start=img_shape[1]-crop_cube_size[1]
                    y_end=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start=k
                    z_end=k+crop_cube_size[2]
                else:
                    z_start=img_shape[2]-crop_cube_size[2]
                    z_end=img_shape[2]
                
                raw_img_crop=raw_img[x_start:x_end, y_start:y_end, z_start:z_end]
                raw_img_crop=raw_img_crop.reshape(1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2])
                raw_img_crop=from_numpy(raw_img_crop).float().to(device)
                
                seg_log_crop=seg_log[x_start:x_end, y_start:y_end, z_start:z_end]
                seg_background_crop=seg_background[x_start:x_end, y_start:y_end, z_start:z_end]
                seg_boundary_crop=seg_boundary[x_start:x_end, y_start:y_end, z_start:z_end]
                seg_foreground_crop=seg_foreground[x_start:x_end, y_start:y_end, z_start:z_end]
                
                with torch.no_grad():
                    seg_crop_output=model(raw_img_crop)
                seg_crop_output_np=seg_crop_output.cpu().detach().numpy()
                
                seg_crop_output_np_bg=seg_crop_output_np[0,0,:,:,:]
                seg_crop_output_np_bd=seg_crop_output_np[0,1,:,:,:]
                seg_crop_output_np_fg=seg_crop_output_np[0,2,:,:,:]
                
                seg_background_temp=np.zeros(seg_background_crop.shape)
                seg_boundary_temp=np.zeros(seg_boundary_crop.shape)
                seg_foreground_temp=np.zeros(seg_foreground_crop.shape)
                
                seg_background_temp[seg_log_crop==1]=(seg_crop_output_np_bg[seg_log_crop==1]+seg_background_crop[seg_log_crop==1])/2
                seg_background_temp[seg_log_crop==0]=seg_crop_output_np_bg[seg_log_crop==0]
                
                seg_boundary_temp[seg_log_crop==1]=(seg_crop_output_np_bd[seg_log_crop==1]+seg_boundary_crop[seg_log_crop==1])/2
                seg_boundary_temp[seg_log_crop==0]=seg_crop_output_np_bd[seg_log_crop==0]
                
                seg_foreground_temp[seg_log_crop==1]=(seg_crop_output_np_fg[seg_log_crop==1]+seg_foreground_crop[seg_log_crop==1])/2
                seg_foreground_temp[seg_log_crop==0]=seg_crop_output_np_fg[seg_log_crop==0]
                
                seg_background[x_start:x_end, y_start:y_end, z_start:z_end]=seg_background_temp
                seg_boundary[x_start:x_end, y_start:y_end, z_start:z_end]=seg_boundary_temp
                seg_foreground[x_start:x_end, y_start:y_end, z_start:z_end]=seg_foreground_temp
                
                seg_log[x_start:x_end, y_start:y_end, z_start:z_end]=1
                
                count=count+1
                
    return {'background': seg_background, 'boundary': seg_boundary, 'foreground': seg_foreground}

def img_3d_erosion_or_expansion(img_3d, kernel_size=3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    org_shape = img_3d.shape
    
    padding = int((kernel_size - 1)/2)
    
    img_3d = img_3d.reshape(1,1,img_3d.shape[0],img_3d.shape[1],img_3d.shape[2])
    img_3d=from_numpy(img_3d).float().to(device)
    
    pool_operation = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=1, padding=padding, dilation=1)
    img_3d = pool_operation(img_3d)
    
    img_3d = torch.nn.functional.interpolate(img_3d, size=org_shape, mode='nearest')
    
    img_3d=img_3d.detach().cpu().numpy()
    img_3d=img_3d.reshape(img_3d.shape[2],img_3d.shape[3],img_3d.shape[4])
    
    return img_3d

### generate super voxels by watershed
'''
def generate_super_vox_by_watershed(input_3d_img, connectivity=10, offset=[1,1,1]):
    input_3d_img_edt=edt.edt(np.array(input_3d_img, dtype=np.uint32, order='F'),black_border=True, order='F',parallel=1)
    return watershed(-input_3d_img_edt, mask=np.array(input_3d_img>0), connectivity=connectivity, offset=offset)
'''
def generate_super_vox_by_watershed(input_3d_img, connectivity=10, min_distance_between_cells = 3):
    input_3d_img_edt=edt.edt(np.array(input_3d_img, dtype=np.uint32, order='F'),
                               black_border=True, order='F',parallel=1)
    coords = peak_local_max(input_3d_img_edt, min_distance=min_distance_between_cells,labels=np.array(input_3d_img>0))
    mask = np.zeros(input_3d_img_edt.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask==True)
    return watershed(-input_3d_img_edt, markers=markers, mask=np.array(input_3d_img>0), connectivity=connectivity)

### cluster on super pixels
def get_outlayer_of_a_3d_shape(a_3d_shape_onehot):
    shape=a_3d_shape_onehot.shape
    
    a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0]-1,:,:]-a_3d_shape_onehot[1:shape[0],:,:]
    a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0]-1,:,:]+a_3d_shape_onehot[1:shape[0],:,:]
    a_3d_crop_diff_y1 = a_3d_shape_onehot[:,0:shape[1]-1,:]-a_3d_shape_onehot[:,1:shape[1],:]
    a_3d_crop_diff_y2 = -a_3d_shape_onehot[:,0:shape[1]-1,:]+a_3d_shape_onehot[:,1:shape[1],:]
    a_3d_crop_diff_z1 = a_3d_shape_onehot[:,:,0:shape[2]-1]-a_3d_shape_onehot[:,:,1:shape[2]]
    a_3d_crop_diff_z2 = -a_3d_shape_onehot[:,:,0:shape[2]-1]+a_3d_shape_onehot[:,:,1:shape[2]]

    outlayer = np.zeros(shape)
    outlayer[1:shape[0],:,:] += np.array(a_3d_crop_diff_x1==1, dtype=np.int8)
    outlayer[0:shape[0]-1,:,:] += np.array(a_3d_crop_diff_x2==1, dtype=np.int8)
    outlayer[:,1:shape[1],:] += np.array(a_3d_crop_diff_y1==1, dtype=np.int8)
    outlayer[:,0:shape[1]-1,:] += np.array(a_3d_crop_diff_y2==1, dtype=np.int8)
    outlayer[:,:,1:shape[2]] += np.array(a_3d_crop_diff_z1==1, dtype=np.int8)
    outlayer[:,:,0:shape[2]-1] += np.array(a_3d_crop_diff_z2==1, dtype=np.int8)
    
    outlayer = np.array(outlayer>0, dtype=np.int8)
    
    return outlayer

def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=2, crop_another_3d_img_by_the_way=None):
    locs = np.where(input_3d_img==val)
    
    shape_of_input_3d_img = input_3d_img.shape
    
    min_x = np.min(locs[0])
    max_x =np.max(locs[0])
    min_y = np.min(locs[1])
    max_y =np.max(locs[1])
    min_z = np.min(locs[2])
    max_z =np.max(locs[2])
    
    x_s = np.clip(min_x-boundary_extend, 0, shape_of_input_3d_img[0])
    x_e = np.clip(max_x+boundary_extend+1, 0, shape_of_input_3d_img[0])
    y_s = np.clip(min_y-boundary_extend, 0, shape_of_input_3d_img[1])
    y_e = np.clip(max_y+boundary_extend+1, 0, shape_of_input_3d_img[1])
    z_s = np.clip(min_z-boundary_extend, 0, shape_of_input_3d_img[2])
    z_e = np.clip(max_z+boundary_extend+1, 0, shape_of_input_3d_img[2])
    
    #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e))
    
    crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    if crop_another_3d_img_by_the_way is not None:
        assert input_3d_img.shape == crop_another_3d_img_by_the_way.shape
        crop_another_3d_img = crop_another_3d_img_by_the_way[x_s:x_e,y_s:y_e,z_s:z_e]
        return crop_3d_img,crop_another_3d_img
    else:
        return crop_3d_img

class Cluster_Super_Vox():
    def __init__(self, min_touching_area=50, min_touching_percentage=0.5, boundary_extend=2):
        super(Cluster_Super_Vox, self).__init__
        self.min_touching_area = min_touching_area
        self.min_touching_percentage = min_touching_percentage
        
        self.boundary_extend = boundary_extend
        
        self.UN_PROCESSED = 0
        self.LONELY_POINT = -1
        self.A_LARGE_NUM = 100000000
        
    def fit(self, input_3d_img, restrict_area_3d=None):
        self.input_3d_img = input_3d_img
        
        if restrict_area_3d is None:
            self.restrict_area_3d = np.array(input_3d_img==0, dtype=np.int8)
        else:
            self.restrict_area_3d = restrict_area_3d
        
        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)
        unique_val_counts = unique_val_counts[unique_vals>0]
        unique_vals = unique_vals[unique_vals>0]
        sort_locs = np.argsort(unique_val_counts)[::-1]
        self.unique_vals = unique_vals[sort_locs]
        
        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED
        
        self.val_outlayer_area = dict()
        for idx, unique_val in enumerate(self.unique_vals):
            print("get val_outlayer area of all vals: "+str(idx/len(self.unique_vals)), end="\r")
            self.val_outlayer_area[unique_val] = self.A_LARGE_NUM
        
        for idx, current_val in enumerate(self.unique_vals):
            print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val), end="\r")
            if self.val_labels[current_val]!=self.UN_PROCESSED:
                continue
            valid_neighbor_vals = self.regionQuery(current_val)
            if len(valid_neighbor_vals)>0:
                print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(valid_neighbor_vals), end="\r")
                self.val_labels[current_val] = current_val
                self.growCluster(valid_neighbor_vals, current_val)
            else:
                self.val_labels[current_val] = self.LONELY_POINT
        
        self.output_3d_img = self.input_3d_img
    
    def fit_V2(self, input_3d_img, restrict_area_3d=None):
        self.input_3d_img = input_3d_img
        
        if restrict_area_3d is None:
            self.restrict_area_3d = np.array(input_3d_img==0, dtype=np.int8)
        else:
            self.restrict_area_3d = restrict_area_3d
        
        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)
        unique_val_counts = unique_val_counts[unique_vals>0]
        unique_vals = unique_vals[unique_vals>0]
        sort_locs = np.argsort(unique_val_counts)[::-1]
        self.unique_vals = unique_vals[sort_locs]
        
        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED
        
        self.val_outlayer_area = dict()
        for idx, unique_val in enumerate(self.unique_vals):
            print("get val_outlayer area of all vals: "+str(idx/len(self.unique_vals)), end="\r")
            self.val_outlayer_area[unique_val] = self.get_outlayer_area(unique_val)
        
        for idx, current_val in enumerate(self.unique_vals):
            print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val), end="\r")
            if self.val_labels[current_val]!=self.UN_PROCESSED:
                continue
            valid_neighbor_vals = self.regionQuery(current_val)
            if len(valid_neighbor_vals)>0:
                print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(valid_neighbor_vals), end="\r")
                self.val_labels[current_val] = current_val
                self.growCluster(valid_neighbor_vals, current_val)
            else:
                self.val_labels[current_val] = self.LONELY_POINT
        
        self.output_3d_img = self.input_3d_img
    
    def get_outlayer_area(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        boundary_extend=self.boundary_extend,
                                                                        crop_another_3d_img_by_the_way=self.restrict_area_3d)
        current_crop_img_onehot = np.array(current_crop_img==current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)
        
        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape
        
        current_crop_img_onehot_outlayer[current_restrict_area>0]=0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)
        
        return current_crop_outlayer_area
    
    def regionQuery(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        boundary_extend=self.boundary_extend,
                                                                        crop_another_3d_img_by_the_way=self.restrict_area_3d)
        
        current_crop_img_onehot = np.array(current_crop_img==current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)
        
        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape
        
        current_crop_img_onehot_outlayer[current_restrict_area>0]=0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)
        
        neighbor_vals, neighbor_val_counts = np.unique(current_crop_img[current_crop_img_onehot_outlayer>0], return_counts=True)
        neighbor_val_counts = neighbor_val_counts[neighbor_vals>0]
        neighbor_vals = neighbor_vals[neighbor_vals>0]
        
        print("current_crop_outlayer_area: "+str(current_crop_outlayer_area), end="\r")
        
        valid_neighbor_vals = self.neighborCheck(neighbor_vals, neighbor_val_counts, current_crop_outlayer_area)
        
        print("valid_neighbor_vals: "+str(valid_neighbor_vals), end="\r")
        print("number of valid_neighbor_vals: "+str(len(valid_neighbor_vals)), end="\r")
        
        return valid_neighbor_vals
        
    def neighborCheck(self, neighbor_vals, neighbor_val_counts, current_crop_outlayer_area):
        neighbor_val_counts = neighbor_val_counts[neighbor_vals>0]
        neighbor_vals = neighbor_vals[neighbor_vals>0]
        
        valid_neighbor_vals = []
        
        for idx, neighbor_val in enumerate(neighbor_vals):
            if neighbor_val_counts[idx]>=self.min_touching_area or \
            (neighbor_val_counts[idx]/current_crop_outlayer_area)>=self.min_touching_percentage or \
            (neighbor_val_counts[idx]/self.val_outlayer_area[neighbor_val])>=self.min_touching_percentage:
                print("touching_area: "+str(neighbor_val_counts[idx]), end="\r")
                print("touching_percentage: "+str(neighbor_val_counts[idx]/current_crop_outlayer_area)+\
                      " and "+str(neighbor_val_counts[idx]/self.val_outlayer_area[neighbor_val]), end="\r")
                valid_neighbor_vals.append(neighbor_val)
        
        double_checked_valid_neighbor_vals = []
        for valid_neighbor_val in valid_neighbor_vals:
            if self.val_labels[valid_neighbor_val]==self.UN_PROCESSED or \
            self.val_labels[valid_neighbor_val]==self.LONELY_POINT:
                double_checked_valid_neighbor_vals.append(valid_neighbor_val)
                
        return np.array(double_checked_valid_neighbor_vals)
    
    def growCluster(self, valid_neighbor_vals, current_val):
        valid_neighbor_vals = valid_neighbor_vals[valid_neighbor_vals>0]
        if len(valid_neighbor_vals)>0:
            for idx, valid_neighbor_val in enumerate(valid_neighbor_vals):
                self.val_labels[valid_neighbor_val]=current_val
                self.input_3d_img[self.input_3d_img==valid_neighbor_val]=current_val
            new_valid_neighbor_vals = self.regionQuery(current_val)
            print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(new_valid_neighbor_vals), end="\r")
            self.growCluster(new_valid_neighbor_vals, current_val)
        else:
            return

# assign boudary voxels to cells with watershed (much faster)
def assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cells, seg_boundary, seg_background=None, compactness=1):
    marker = seg_foreground_single_cells
    
    distance_of_boundary=np.array(seg_boundary, dtype=np.uint32, order='F')
    distance_of_boundary=edt.edt(distance_of_boundary,black_border=True, order='F',parallel=1)
    
    if seg_background is not None:
        mask = np.array(seg_background==0)
        distance_of_remaining_foreground = 1 - seg_background - np.array(seg_foreground_single_cells>0, dtype=np.float)
    else:
        mask = np.ones(seg_foreground_single_cells.shape)
        distance_of_remaining_foreground = 1 - np.array(seg_foreground_single_cells>0, dtype=np.float)
    distance_of_remaining_foreground=np.array(distance_of_remaining_foreground, dtype=np.uint32, order='F')
    distance_of_remaining_foreground=edt.edt(distance_of_remaining_foreground,black_border=True, order='F',parallel=1)

    distance = distance_of_remaining_foreground+np.max(distance_of_remaining_foreground)*distance_of_boundary
    distance=np.min(distance)*np.array(seg_foreground_single_cells>0, dtype=np.float)+distance
    
    labels = watershed(distance, marker, mask=mask, compactness=compactness)
    
    return labels

def delete_too_small_cluster(seg_ins, threshold=100):
    unique_vals, counts = np.unique(seg_ins, return_counts=True)
    for unique_val in unique_vals[counts<threshold]:
        seg_ins[seg_ins==unique_val]=0
    return seg_ins

# last step: reassign unique numbers
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