# dataset load and transform
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
import h5py

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio

class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size=(output_size, output_size, output_size)
        else:
            assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size=output_size
        
    def random_crop_start_point(self, input_size):
        assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size
        assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size; input size: "+str(input_size)
        
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size
        
        d_start=np.random.randint(0, d-d_new+1)
        h_start=np.random.randint(0, h-h_new+1)
        w_start=np.random.randint(0, w-w_new+1)
        
        return d_start, h_start, w_start
    
    def __call__(self, img_3d, start_points=None):
        img_3d=np.array(img_3d)
        
        d, h, w=img_3d.shape
        d_new, h_new, w_new=self.output_size
        
        assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size"
        
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)
        
        d_start, h_start, w_start = start_points
        
        crop=img_3d[d_start:d_start+d_new, h_start:h_start+h_new, w_start:w_start+w_new]
        
        return crop

class Normalization_np(object):
    def __init__(self):
        self.name = 'ManualNormalization'
    
    def __call__(self, img_3d):
        img_3d-=np.min(img_3d)
        max_99_val=np.percentile(img_3d, 99)
        if max_99_val>0:
            img_3d = img_3d/max_99_val*255
        return img_3d

class Cell_Seg_3D_Dataset(Dataset):
    def __init__(self, data_dict):
        # each item of data_dict is {name:{"raw":raw img path, "background": background img path,
        # "boundary": boundary img path, "foreground": foreground img path}}
        self.data_dict = data_dict
        self.name_list = np.array(list(data_dict))
        self.para = {}

    def __len__(self):
        return len(self.name_list) 
    
    def __getitem__(self, idx):
        return self.get(idx, file_format=self.para["file_format"], \
            crop_size = self.para["crop_size"], \
                boundary_importance = self.para["boundary_importance"], \
                    need_tensor_output = self.para["need_tensor_output"], \
                        need_transform = self.para["need_transform"])
    
    def set_para(self, file_format='.npy', crop_size = (64, 64, 64), \
        boundary_importance = 1, need_tensor_output = True, need_transform = True):
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["boundary_importance"] = boundary_importance
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    def set_random_crop_size(self, crop_size_range=[32,64]):
        return np.random.randint(crop_size_range[0],crop_size_range[1],size=(3))
    
    def get(self, idx, file_format='.npy', crop_size = (64, 64, 64), \
            boundary_importance = 1, need_tensor_output = True, need_transform = True):
        crop_size=tuple(crop_size)
        # print("random crop size: "+str(crop_size))
        random3dcrop=Random3DCrop_np(crop_size)
        
        normalization=Normalization_np()
        
        name = self.name_list[idx]
        if file_format == ".npy":
            raw_3d_img = np.load(self.data_dict[name]["raw"])
            seg_boundary = np.load(self.data_dict[name]["boundary"])
            seg_foreground = np.load(self.data_dict[name]["foreground"])
            #seg_background = np.load(self.data_dict[name]["background"])
        elif file_format == ".tif":
            raw_3d_img = io.imread(self.data_dict[name]["raw"])                
            seg_boundary = io.imread(self.data_dict[name]["boundary"])
            seg_foreground = io.imread(self.data_dict[name]["foreground"])
            #seg_background = io.imread(self.data_dict[name]["background"])
        elif file_format == ".h5":
            hf = h5py.File(self.data_dict[name], 'r+')
            raw_3d_img = np.array(hf["raw"])
            seg_boundary = np.array(hf["boundary"])
            seg_foreground = np.array(hf["foreground"])
            hf.close()
        elif file_format == ".npz":
            npz_file = np.load(self.data_dict[name])
            raw_3d_img = np.array(npz_file["raw"])
            seg_boundary = np.array(npz_file["boundary"])
            seg_foreground = np.array(npz_file["foreground"])
                
        raw_3d_img = np.nan_to_num(raw_3d_img, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        seg_boundary = np.nan_to_num(seg_boundary, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        seg_foreground = np.nan_to_num(seg_foreground, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        seg_background = np.array((seg_boundary+seg_foreground)==0, dtype=np.int)
            
        #raw_3d_img=normalization(raw_3d_img)
                
        raw_3d_img = np.array(raw_3d_img, float)
        seg_background = np.array(seg_background, float)
        seg_boundary = np.array(seg_boundary, float)
        seg_foreground = np.array(seg_foreground, float)
            
        assert raw_3d_img.shape == seg_background.shape
        assert seg_background.shape == seg_boundary.shape
        assert seg_boundary.shape == seg_foreground.shape
            
        start_points=random3dcrop.random_crop_start_point(raw_3d_img.shape)
        raw_3d_img=random3dcrop(raw_3d_img, start_points=start_points)
        seg_background=random3dcrop(seg_background, start_points=start_points)
        seg_boundary=random3dcrop(seg_boundary, start_points=start_points)
        seg_foreground=random3dcrop(seg_foreground, start_points=start_points)
        
        raw_3d_img = np.expand_dims(raw_3d_img, axis=0)
        seg_background = np.expand_dims(seg_background, axis=0)
        seg_boundary = np.expand_dims(seg_boundary, axis=0)
        seg_foreground = np.expand_dims(seg_foreground, axis=0)

        output = {'raw': raw_3d_img, 'background': seg_background, 'boundary': seg_boundary, 'foreground': seg_foreground}
        
        output.update(self.get_weights(output, boundary_importance))
        
        if need_tensor_output:
            output = self.to_tensor(output)

            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)
        
        return output
    
    def get_weights(self, images, boundary_importance): # images: a dict, each item should be in numpy.array format
        seg_background=images['background']
        seg_boundary=images['boundary']*boundary_importance # boundary is boundary_importance times more important than others
        seg_foreground=images['foreground']
        
        seg_background_zeros=np.array(seg_background==0, dtype=int)*0.5
        seg_boundary_zeros=np.array(seg_boundary==0, dtype=int)*0.5
        seg_foreground_zeros=np.array(seg_foreground==0, dtype=int)*0.5
        
        return {'weights_background': seg_background+seg_background_zeros, 'weights_boundary': seg_boundary+seg_boundary_zeros, 'weights_foreground': seg_foreground+seg_foreground_zeros}

    def to_tensor(self, images):
        images_tensor={}
        for item in images.keys():
            images_tensor[item]=from_numpy(images[item]).float()
        return images_tensor
    
    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}
        
        for item in image_tensors.keys():
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob)])#,tio.RandomAffine(p=prob)])
        subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),
            tio.RandomNoise(p=prob),tio.RandomMotion(p=prob),tio.RandomBiasField(p=prob),tio.RandomSpike(p=prob),tio.RandomGhosting(p=prob)])
        subject_all_imgs['raw'] = transform_val(subject_all_imgs['raw'])
        
        for item in subject_all_imgs.keys():
            image_tensors[item] = subject_all_imgs[item].data
        
        return image_tensors

from medpy.io import load
def show_unpreprocessed_images(raw_img_file_path, seg_img_file_path):
    raw_img_names=os.listdir(raw_img_file_path)
    raw_img_names.sort()
    seg_img_names=os.listdir(seg_img_file_path)
    seg_img_names.sort()
    assert raw_img_names==seg_img_names
    img_names=raw_img_names
    del raw_img_names
    del seg_img_names
    k=np.random.randint(0, len(img_names)-1)
    
    raw_img, raw_h = load(raw_img_file_path+'/'+img_names[k])
    seg_img, seg_h = load(seg_img_file_path+'/'+img_names[k])
    
    raw_img=np.array(raw_img)
    seg_img=np.array(seg_img)
    assert raw_img.shape==seg_img.shape
    
    N=np.int(np.round(raw_img.shape[2]/2))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Raw image')
    #plt.axis('off')
    plt.imshow(raw_img[:, :, N])
    
    plt.subplot(1, 3, 2)
    plt.title('Segmentation')
    #plt.axis('off')
    plt.imshow(seg_img[:, :, N])
    
    plt.subplot(1, 3, 3)
    plt.title('Raw image & Segmentation')
    #plt.axis('off')
    plt.imshow(raw_img[:, :, N])
    plt.imshow(seg_img[:, :, N], alpha=0.5)

if __name__=="__main__":
    import pickle
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    HMS_dataset_info = load_obj("../dataset_info/HMS_dataset_info")
    dataset = Cell_Seg_3D_Dataset(HMS_dataset_info['train'])
    idx = 0
    crop_size = dataset.set_random_crop_size()
    output = dataset.get(idx, file_format='.npy', crop_size = crop_size, \
        boundary_importance = 1, need_tensor_output = True, need_transform = True)
    for item in output.keys():
        print(item, output[item].shape)
    dataset.set_para(file_format='.npy', crop_size = (64, 64, 64), \
        boundary_importance = 1, need_tensor_output = True, need_transform = True)
    num_workers = 4
    Dataset_loader = DataLoader(dataset, batch_size=5, shuffle=True, \
        num_workers=num_workers, pin_memory=False, persistent_workers=False)#(num_workers > 1))
    batch = next(iter(Dataset_loader))
    for item in batch.keys():
        print(item, batch[item].shape)