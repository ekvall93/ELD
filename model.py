import sys

import glob, os, sys, csv, math, functools, collections, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, vgg16_bn, vgg11_bn
from MTFAN import FAN, convertLayer, GeoDistill
from utils import *
import itertools
from torch.autograd import Variable
from torchgeometry.contrib import spatial_soft_argmax2d
import kornia

import random


from kornia.geometry.conversions import normalize_homography
from kornia.utils.helpers import _torch_inverse_cast

from kornia.geometry.linalg import transform_points

from warp import WARP
from piqa import MS_SSIM

def crop(img1, img2):
    n = img1.size(1)
    G = img1.mean(1, keepdim=True)
    black_mask_Y_perm = (G < 0.1)
    img2 = torch.where(black_mask_Y_perm.repeat(1,n,1,1), torch.zeros_like(img2), img2)
    

    #G = self.grayscale_transform(self.A['ImP'][:-1])
    G = img2.mean(1, keepdim=True)
    black_mask = (G < 0.1)
    img1 = torch.where(black_mask.repeat(1,n,1,1), torch.zeros_like(img1), img1)

    return img1, img2

class HomoModel():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10,                option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, n_chanels = 3, elastic=None, warp='tps'):
        self.npoints = npts
        self.gradclip = gradclip
        self.warmup_steps = warmup_steps
        self.n_chanels = n_chanels
        

        self.samples = 0
        
        # - define FAN
        
        self.FAN = FAN(1,n_points=self.npoints, in_channels=self.n_chanels)


        #path_to_check = "/home/markus.ekvall/landmarks/Exp_229/model_8.fan.pth"
        if not option == 'scratch':
            net_dict = self.FAN.state_dict()
            pretrained_dict = torch.load(path_to_check, map_location='cuda')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
            net_dict.update(pretrained_dict)
            self.FAN.load_state_dict(net_dict, strict=True)
            if option == 'incremental':
                print('Option is incremental')
                self.FAN.apply(convertLayer)

        if not option == 'scratch':
            net_dict = self.GEN.state_dict()
            pretrained_dict = torch.load(path_to_check, map_location='cuda')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
            net_dict.update(pretrained_dict)
            self.GEN.load_state_dict(net_dict, strict=True)
            if option == 'incremental':
                print('Option is incremental')
                self.GEN.apply(convertLayerGen)

        self.warp = WARP(warp)

        self.mssi = MS_SSIM(window_size=5,padding=True, n_channels=self.n_chanels).cuda()
        
        # - multiple GPUs
        if torch.cuda.device_count() > 1:
            self.FAN = torch.nn.DataParallel(self.FAN)
              
        self.FAN.to('cuda').train()
      
        # - VGG for perceptual loss
        self.loss_network = LossNetwork(torch.nn.DataParallel(vgg16_bn(pretrained=True))) if torch.cuda.device_count() > 1 else LossNetwork(vgg16(pretrained=True))
        self.loss_network.eval()
        self.loss_network.to('cuda')
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None
        
        # - define losses for reconstruction

    def _resume(self,path_fan, path_gen):
        self.FAN.load_state_dict(torch.load(path_fan))

    def _save(self, path_to_models, epoch):
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
        
    def _set_batch(self,data):
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}


    def forward(self, n_iter):
        b = self.A['Im'].shape[0]
        self.samples += b
        

        self.FAN.zero_grad()
       
        b = self.A['Im'].shape[0]
        rand_ix = list(range(b))
        random.shuffle(rand_ix)

        random.shuffle(rand_ix)

        X = torch.cat([self.A['Im'], self.A['ImP']],0)
        H_X = self.FAN(X)
        H, H_P = torch.split(H_X, [b, b], dim=0)
        
        Pts = 4 * spatial_soft_argmax2d(H, False)
        Pts_P = 4 * spatial_soft_argmax2d(H_P, False)
        
        _Pts, _Pts_P = Pts, Pts_P
  
        X = self.A['ImP']            
       
        X = torch.cat([X, X[rand_ix]],0)
        _Pts = torch.cat([_Pts, _Pts_P],0)
        _Pts_P = torch.cat([_Pts_P, _Pts_P[rand_ix]],0)

        
        X = self.warp.warp_img(X, _Pts_P, _Pts)
        
        X, Imgs = crop(X,  torch.cat([self.A['Im'], self.A['ImP']],0))
        X, Y = torch.split(X, [b, b], dim=0)
        self.A['Im'], self.A['ImP'] = torch.split(Imgs, [b, b], dim=0)

        R_loss = (1 - self.mssi(X, self.A['Im']))
        
        consistent_loss = (1 - self.mssi(Y, self.A['ImP']))
        
        perp_loss =  R_loss + 0.1 * consistent_loss
     
        self.loss['rec'] = perp_loss
        self.loss['perp'] = perp_loss

        self.loss['rec'].backward()
        
        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
            
   
        return {'Heatmap' : H_P, 
                'generated': X, 
                'Pts' : Pts,
                'Pts_P' : Pts_P,
                'genSamples': Y
                }

