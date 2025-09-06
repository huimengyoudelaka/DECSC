import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class Conv3d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv3d_cd, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_out c_in k1 k2 k3 -> c_out c_in (k1 k2 k3)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3**3).fill_(0).to(conv_weight)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 13] = conv_weight[:, :, 13] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_out c_in (k1 k2 k3) -> c_out c_in k1 k2 k3', k1=conv_shape[2], k2=conv_shape[3], k3=conv_shape[4])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv3d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv3d_ad, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_out c_in k1 k2 k3 -> c_out c_in (k1 k2 k3)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [12,9,0,15,3,1,6,7,4,21,18,10,24,13,2,16,8,5,22,19,20,25,23,11,26,17,14]]
        conv_weight_ad = Rearrange('c_out c_in (k1 k2 k3) -> c_out c_in k1 k2 k3', k1=conv_shape[2], k2=conv_shape[3], k3=conv_shape[4])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias

class Conv3d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv3d_hd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3**3).fill_(0).to(conv_weight)
        conv_weight_hd[:, :, [0, 3, 6, 9, 12, 15, 18, 21, 24]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8, 11, 14, 17, 20, 23, 26]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_out c_in (k1 k2 k3) -> c_out c_in k1 k2 k3', k1=3, k2=3, k3=3)(conv_weight_hd)
        return conv_weight_hd, self.conv.bias

class Conv3d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv3d_vd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3**3).fill_(0).to(conv_weight)
        conv_weight_vd[:, :, [0, 1, 2, 9, 10, 11, 18, 19, 20]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8, 15, 16, 17, 24, 25, 26]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_out c_in (k1 k2 k3) -> c_out c_in k1 k2 k3', k1=3, k2=3, k3=3)(conv_weight_vd)
        return conv_weight_vd, self.conv.bias

class Conv3d_bd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv3d_bd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
        conv_weight_bd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3**3).fill_(0).to(conv_weight)
        conv_weight_bd[:, :, :9] = conv_weight[:, :, :]
        conv_weight_bd[:, :, 18:] = -conv_weight[:, :, :]
        conv_weight_bd = Rearrange('c_out c_in (k1 k2 k3) -> c_out c_in k1 k2 k3', k1=3, k2=3, k3=3)(conv_weight_bd)
        return conv_weight_bd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__() 
        self.conv1_1 = Conv3d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv3d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv3d_vd(dim, dim, 3, bias=True)
        # self.conv1_4 = Conv3d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv3d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        # w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w5
        b = b1 + b2 + b3 + b5
        res = nn.functional.conv3d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res

class DEConv2(nn.Module):
    def __init__(self, dim):
        super(DEConv2, self).__init__() 
        self.conv1_1 = Conv3d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv3d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv3d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv3d_bd(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv3d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv3d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res