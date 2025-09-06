# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import dataloaders_hsi_test
from utility import *
from hsi_setup import Engine, train_options
import models
import  scipy.io as scio
from ops.utils_blocks import block_module
import numpy as np
from showRes import saveAsCsv

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)
    ###modified###
    basefolder = opt.testroot
    # for noise in (15, 55, 95):
    noise=opt.noise_level
    # if noise == "Corr":
    #     test_path = "/data_3/yejin/test_dataset/ICVL/corr"
    # else:
    #     test_path = os.path.join(basefolder, str(noise))
    test_path = os.path.join(basefolder, str(noise))
    # test_path = "/data/yejin/projects/SVD/temp"
    # params = {
    #     'crop_out_blocks': 0,
    #     'ponderate_out_blocks': 1,
    #     'sum_blocks': 0,
    #     'pad_even': 1,  # otherwise pad with 0 for las
    #     'centered_pad': 0,  # corner pixel have only one estimate
    #     'pad_block': 1,  # pad so each pixel has S**2 estimate
    #     'pad_patch': 0,
    #     # pad so each pixel from the image has at least S**2 estimate from 1 block
    #     'no_pad': 0,
    #     'custom_pad': None,
    #     'avg': 1}
    # block = block_module(opt.patch_size, opt.stride_test, opt.kernel_size, params)
    print('noise:   ', noise, end='')
    test = dataloaders_hsi_test.get_dataloaders([test_path], verbose=True, grey=False)
    patch_size = 128
    _, _, res_arr = engine.validate_patch2(test['test'], '', (31,patch_size,patch_size), (31,patch_size,patch_size), (0,0,0))
    # _, _, res_arr = engine.validate(test['test'], '')
    # res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
    # print(res_arr.mean(axis=0))
    # _, _, res_arr = engine.validate(test['test'], '',block,opt.batchSize)
    aver = [np.mean(res_arr, axis=0)]
    # savePath = './res_95/SVD40/100/'
    saveName = opt.resumePath.split('/')
    # savePath = './houston/res_'+str(noise)+'/'+saveName[-2]
    savePath = './icvl/res_'+str(noise)+'/'+saveName[-2]
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savePath = savePath +'/'+saveName[-1].split('.')[0]+'_'
    saveAsCsv(savePath+str(noise) + 'dB_' + 'Res.csv',res_arr)
    saveAsCsv(savePath+str(noise) + 'dB_' + 'ResAver.csv',aver)
    # scio.savemat('./res/'+str(noise) + 'dB_' + 'Svd5Res.mat',
    #              {'res_arr': res_arr})
    # for noise in (15,55,95):
    #     test_path = os.path.join(basefolder, str(noise)+'dB/')
    #     print('noise:   ',noise,end='')
    #     test = dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)
    #
    #     # res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
    #     # print(res_arr.mean(axis=0))
    #     _,_,res_arr=engine.validate(test['test'], '')

    ###modified###
