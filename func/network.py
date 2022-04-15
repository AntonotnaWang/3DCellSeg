# model
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class ResModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1):
        super(ResModule, self).__init__()
        self.batchnorm_module=nn.BatchNorm3d(num_features=in_channels)
        self.conv_module=nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        h=F.relu(self.batchnorm_module(x))
        h=self.conv_module(h)
        return h+x

class CellSegNet_basic_lite(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func = "softmax"):
        super(CellSegNet_basic_lite, self).__init__()
        
        self.conv1=nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2=nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1=nn.BatchNorm3d(num_features=32)
        self.conv3=nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1=ResModule(64, 64)
        self.conv4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2=ResModule(64, 64)
        self.conv5=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3=ResModule(64, 64)
        
        self.deconv1=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2=nn.BatchNorm3d(num_features=64)
        self.deconv2=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3=nn.BatchNorm3d(num_features=64)
        self.deconv3=nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4=nn.BatchNorm3d(num_features=32)
        self.conv6=nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)
        
        self.output_func = output_func
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))
        
        h = self.conv3(c1)
        c2 = self.resmodule1(h)
        
        h = self.conv4(c2)
        c3 = self.resmodule2(h)
        
        h = self.conv5(c3)
        c4 = self.resmodule3(h)
        
        c4 = self.deconv1(c4)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape=c3.shape
        delta_c4_x=int(np.floor((c4.shape[2]-c3_shape[2])/2))
        delta_c4_y=int(np.floor((c4.shape[3]-c3_shape[3])/2))
        delta_c4_z=int(np.floor((c4.shape[4]-c3_shape[4])/2))
        c4 = c4[:, :,
                delta_c4_x:c3_shape[2]+delta_c4_x,
                delta_c4_y:c3_shape[3]+delta_c4_y,
                delta_c4_z:c3_shape[4]+delta_c4_z]
        
        h = c4 + c3
        
        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape=c2.shape
        delta_c2_2_x=int(np.floor((c2_2.shape[2]-c2_shape[2])/2))
        delta_c2_2_y=int(np.floor((c2_2.shape[3]-c2_shape[3])/2))
        delta_c2_2_z=int(np.floor((c2_2.shape[4]-c2_shape[4])/2))
        c2_2 = c2_2[:, :,
                delta_c2_2_x:c2_shape[2]+delta_c2_2_x,
                delta_c2_2_y:c2_shape[3]+delta_c2_2_y,
                delta_c2_2_z:c2_shape[4]+delta_c2_2_z]
        
        h = c2_2 + c2
        
        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape=c1.shape
        delta_c1_2_x=int(np.floor((c1_2.shape[2]-c1_shape[2])/2))
        delta_c1_2_y=int(np.floor((c1_2.shape[3]-c1_shape[3])/2))
        delta_c1_2_z=int(np.floor((c1_2.shape[4]-c1_shape[4])/2))
        c1_2 = c1_2[:, :,
                delta_c1_2_x:c1_shape[2]+delta_c1_2_x,
                delta_c1_2_y:c1_shape[3]+delta_c1_2_y,
                delta_c1_2_z:c1_shape[4]+delta_c1_2_z]
        
        h = c1_2 + c1
        
        h = self.conv6(h)
        
        output = F.softmax(h, dim=1)
        
        return output
    
class VoxResNet(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func = "softmax"):
        super(VoxResNet, self).__init__()
        
        self.conv1a=nn.Conv3d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1a=nn.BatchNorm3d(num_features=32)
        self.conv1b=nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1b=nn.BatchNorm3d(num_features=32)
        self.conv1c=nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res2=ResModule(64, 64)
        self.res3=ResModule(64, 64)
        self.bnorm3=nn.BatchNorm3d(num_features=64)
        self.conv4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res5=ResModule(64, 64)
        self.res6=ResModule(64, 64)
        self.bnorm6=nn.BatchNorm3d(num_features=64)
        self.conv7=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res8=ResModule(64, 64)
        self.res9=ResModule(64, 64)
        
        self.c1deconv=nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.c1conv=nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, padding=1)
        self.c2deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.c2conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        self.c3deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=6, stride=4, padding=1)
        self.c3conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        self.c4deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=10, stride=8, padding=1)
        self.c4conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        
        self.output_func = output_func
    def forward(self, x):
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h))
        h = self.conv1b(h)
        c1 = F.relu6(self.c1deconv(h))
        c1 = self.c1conv(c1)
        
        h = F.relu(self.bnorm1b(h))
        h = self.conv1c(h)
        h = self.res2(h)
        h = self.res3(h)
        c2 = F.relu6(self.c2deconv(h))
        c2 = self.c2conv(c2)
        
        h = F.relu(self.bnorm3(h))
        h = self.conv4(h)
        h = self.res5(h)
        h = self.res6(h)
        c3 = F.relu6(self.c3deconv(h))
        c3 = self.c3conv(c3)
        
        h = F.relu(self.bnorm6(h))
        h = self.conv7(h)
        h = self.res8(h)
        h = self.res9(h)
        c4 = F.relu6(self.c4deconv(h))
        c4 = self.c4conv(c4)
        
        c = c1 + c2 + c3 + c4
        
        output = F.softmax(c, dim=1)
        
        return output