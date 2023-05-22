# The following implementation is based on the techniques described in:
# "Object Landmark Discovery Through Unsupervised Adaptation" by Enrique Sanchez and Georgios Tzimiropoulos
# You can find the article here: http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf
#
# For more details on the practical implementation of these techniques, check out the corresponding GitHub repository:
# https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019

import sys

import glob, os, sys, csv, math, functools, collections, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, vgg16_bn, vgg11_bn
from MTFAN import FAN, MultiModalFAN, convertLayer
from utils import *
import itertools
from torch.autograd import Variable
from torchgeometry.contrib import spatial_soft_argmax2d


import random

from warp import WARP, Rigid
from piqa import MS_SSIM


def loadFan(npoints=10,n_channels=3,path_to_model=None, path_to_core=None):
    net = FAN(1,in_channels=n_channels, n_points=npoints).to('cuda')
    checkpoint = torch.load(path_to_model)
    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
    if path_to_core is not None:
        net_dict = net.state_dict()
        pretrained_dict = torch.load(path_to_core, map_location='cuda')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict, strict=True)
        net.apply(convertLayer)
    net.load_state_dict(checkpoint)
    return net.to('cuda')

def loadMultiModalFan(npoints=10,n_channels=3,path_to_model=None, path_to_core=None):
    net = MultiModalFAN(1,in_channels=n_channels, n_points=npoints).to('cuda')
    checkpoint = torch.load(path_to_model)
    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
    if path_to_core is not None:
        net_dict = net.state_dict()
        pretrained_dict = torch.load(path_to_core, map_location='cuda')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict, strict=True)
        net.apply(convertLayer)
    net.load_state_dict(checkpoint)
    return net.to('cuda')


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

class UniModal():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10, option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, n_chanels = 3, warp='tps', crop=True):
        self.npoints = npts
        self.gradclip = gradclip
        self.n_chanels = n_chanels
        
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
      
        
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None

        self.mask = np.ones(self.npoints).astype(bool)

        self.crop = crop
        
        # - define losses for reconstruction

    def _resume(self,path_fan, path_gen):
        self.FAN.load_state_dict(torch.load(path_fan))

    def _save(self, path_to_models, epoch):
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
        
    def _set_batch(self,data):
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}

    def new_mask(self):
        self.mask = np.random.choice(2, self.npoints, p=[0.1, 0.9]).astype(bool)

    def set_mask(self, p=0.5):
        if random.uniform(0,1)>p:
            self.new_mask()
            if random.uniform(0,1)>p:
                self.new_mask()
            else:
                self.mask = np.ones(self.npoints).astype(bool)

    def forward(self, n_iter):
        b = self.A['Im'].shape[0]
        
        

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
        self.set_mask()
        _Pts, _Pts_P = Pts[:,self.mask], Pts_P[:,self.mask]
  
        X = self.A['ImP']            
       
        X = torch.cat([X, X[rand_ix]],0)
        _Pts = torch.cat([_Pts, _Pts_P],0)
        _Pts_P = torch.cat([_Pts_P, _Pts_P[rand_ix]],0)

        
        X = self.warp.warp_img(X, _Pts_P, _Pts)
        
        if self.crop:
            X, Imgs = crop(X,  torch.cat([self.A['Im'], self.A['ImP']],0))
        else:
            X, Imgs = X,  torch.cat([self.A['Im'], self.A['ImP']],0)


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


class MultiModal():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10,                option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, n_chanels = 3, warp='tps', crop=True):
        self.npoints = npts
        self.gradclip = gradclip
        
        
        self.n_chanels = n_chanels

        self.FAN = MultiModalFAN(1,n_points=self.npoints)
        


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
        
        
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None
        
        # - define losses for reconstruction
        self.SelfLoss = torch.nn.MSELoss().to('cuda')
        self.step = 0

        self.new_mask()

        self.crop = crop

                                                                                                   
    def _resume(self,path_fan, path_gen):
        self.FAN.load_state_dict(torch.load(path_fan))
               
    def _save(self, path_to_models, epoch):
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
       
    def _set_batch(self,data):
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}

    def const(self,anneal_start=0, anneal_end=1_000):
        return np.clip((self.step - anneal_start)/(anneal_end - anneal_start), 0.0, 1.0)

    def new_mask(self):
        self.mask = np.random.choice(2, self.npoints, p=[0.1, 0.9]).astype(bool)

    def set_mask(self, p=0.5):
        if random.uniform(0,1)>p:
            self.new_mask()
            if random.uniform(0,1)>p:
                self.new_mask()

        if np.sum(self.mask) == self.npoints:
            self.mask[np.random.randint(self.npoints)] = False


    def forward(self, n_iter):
        

        self.A['mod'] = self.A['mod'].squeeze()
        
        self.step +=1 
        b = self.A['Im'].shape[0]
        
        self.FAN.zero_grad()
        
        b = self.A['Im'].shape[0]
        rand_ix = list(range(b))
        random.shuffle(rand_ix)

        X = torch.cat([self.A['Im'], self.A['ImP']],0)
        MOD = torch.cat([self.A['mod'], self.A['mod']],0)
        H_X, A =  self.FAN(X, MOD == 1)
        
        H, H_P = torch.split(H_X, [b, b], dim=0)

        #Get Im's activations
        A = [a[:b] for a in A]
        
        Pts = 4 * spatial_soft_argmax2d(H, False)
        Pts_P = 4 * spatial_soft_argmax2d(H_P, False)

        self.set_mask()

        _Pts, _Pts_P = Pts, Pts_P


        _Pts, _Pts_P = _Pts[:,self.mask], _Pts_P[:,self.mask]

        Y = self.warp.warp_img(self.A['ImP'], _Pts_P, _Pts)  

        if self.crop:     
            Y, Im = crop(Y, self.A['Im'].clone())
        else:
            Y, Im = Y, self.A['Im'].clone()

        R_loss =  (1 - self.mssi(Y, Im))

       
        Z, mod = self.warp.warp_img(self.A['ImP'][rand_ix], _Pts_P[rand_ix], _Pts), self.A['mod'][rand_ix]
        H_Z, A_Z = self.FAN(Z, mod == 1)

        l = 0
        for i in range(len(A)):
            l += self.SelfLoss(A_Z[i], A[i])
            
            if i == 0:
                #only use thist layers activations
                break
        
        l = l / (i + 1)




        X_2 = self.A['ImP']    
        ixs = self.A['mod'] == 0


        X_mod1 = X_2[ixs]
        X_mod2 = X_2[~ixs]
        _Pts_P_mod1 = _Pts_P[ixs]
        _Pts_P_mod2 = _Pts_P[~ixs]

        #make random ix for X_mod1 and X_mod2
        rand_ix_m1 = list(range(len(X_mod1)))
        random.shuffle(rand_ix)
        
        rand_ix_m2 = list(range(len(X_mod2)))
        random.shuffle(rand_ix)

        X_mod1_ = self.warp.warp_img(X_mod1[rand_ix_m1], _Pts_P_mod1[rand_ix_m1], _Pts_P_mod1)
        X_mod2_ = self.warp.warp_img(X_mod2[rand_ix_m2], _Pts_P_mod2[rand_ix_m2], _Pts_P_mod2)
        
        if self.crop:
            X_mod1, X_mod1_ = crop(X_mod1, X_mod1_)
            X_mod2_, X_mod2 = crop(X_mod2_, X_mod2)

        inter_mod_loss = 0.5 * ((1 - self.mssi(X_mod1, X_mod1_)) + (1 - self.mssi(X_mod2_, X_mod2)))

        perp_loss = R_loss +  10 * self.const(anneal_end=500) * l + 0.1 * inter_mod_loss
       

        self.loss['rec'] = perp_loss
        self.loss['perp'] = perp_loss

        self.loss['rec'].backward()
        
        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
            
        return {'Heatmap' : H_P, 
                'generated': Y, 
                'Pts' : Pts,
                'Pts_P' : Pts_P,
                'genSamples': H_Z
                }

def toGrey(img):
    #use dot product to convert to greyscale
    return torch.einsum('bchw, c -> bhw', img, torch.tensor([0.299, 0.587, 0.114]).cuda()).unsqueeze(1)

def compute_area(img):
    return (img.mean(1) > 0.1).flatten(1,2).sum(1)

def get_fraction_of_black(img1):
    G = img1
    black_mask_Y_perm = (G < 0.01)
    return black_mask_Y_perm.squeeze().flatten(1,2).sum(1) / (img1.shape[2] * img1.shape[3])

class Model3D():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10,                option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, hyper=False, n_genes = None, warp='tps', n_chanels=3, crop=True):
        self.npoints = npts
        self.gradclip = gradclip
        self.warmup_steps = warmup_steps
        self.n_genes = n_genes
        self.train_fan = True
        self.train_tps = True

        self.samples = 0
        
        # - define FAN
        self.warmpup_finnished = False
        self.hyper = hyper
        self.n_sucess = 0

       
        self.FAN = FAN(1,n_points=self.npoints)

        self.size = size


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

      
        self.TPS = None
        self.warp = WARP(warp)
        
        self.mssi = MS_SSIM(window_size=5, reduction='none', n_channels=1).cuda()
            
        self.FAN.to('cuda').train()
        
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None
        
        # - define losses for reconstruction
        self.SelfLoss = torch.nn.MSELoss().to('cuda')
        
        self.step = 0

        self.new_mask()
        self.rigid = Rigid()

        self.crop = crop


    def _resume(self,path_fan, path_gen):
        self.FAN.load_state_dict(torch.load(path_fan))
               
    def _save(self, path_to_models, epoch):
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')

    def _set_batch(self,data):
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}

    def const(self,anneal_start=0, anneal_end=1_000):
        return np.clip((self.step - anneal_start)/(anneal_end - anneal_start), 0.0, 1.0)

    def new_mask(self):
        self.mask = np.random.choice(2, self.npoints, p=[0.2, 0.8]).astype(bool)
        if np.sum(self.mask) == self.npoints:
                self.mask[np.random.randint(0,self.npoints)] = False

    def set_mask(self, p=0.5):
        if random.uniform(0,1)>p:
            self.new_mask()
            if random.uniform(0,1)>p:
                self.new_mask()
            else:
                self.mask = np.ones(self.npoints).astype(bool)
                self.mask[np.random.randint(0,self.npoints)] = False

    
    def forward(self, n_iter):
        _, indices = torch.sort(self.A['idx'].squeeze())
        self.A['Im'] = self.A['Im'][indices]
        self.A['ImP'] = self.A['ImP'][indices]
        self.A['ImPD'] = self.A['ImPD'][indices]
        self.A['idx'] = self.A['idx'][indices]

        self.step +=1 
        b = self.A['Im'].shape[0]
        self.samples += b
        

        self.FAN.zero_grad()
        
        b = self.A['Im'].shape[0]
        rand_ix = list(range(b))
        random.shuffle(rand_ix)
       

        random.shuffle(rand_ix)


        #pick a random number between 0 and 20
        random_number = random.randint(0, 30)
        

        X = torch.cat([self.A['Im'], self.A['ImP']],0)
        H_X = self.FAN(X)
        H, H_P = torch.split(H_X, [b, b], dim=0)
        
        
        Pts = 4 * spatial_soft_argmax2d(H, False)
        Pts_P = 4 * spatial_soft_argmax2d(H_P, False)

        step = 4
        rand_ix = [random.randint(max(0, i-step), min(b-1, i+step)) for i in range(b)]
        step = 50
    
        
        rand_ix_large = [random.randint(0, i) for i in range(b)]


        self.set_mask()
        _Pts, _Pts_P = Pts, Pts_P
        _Pts_missing, _Pts_P_missing = _Pts[:,~self.mask], _Pts_P[:,~self.mask]
        _Pts, _Pts_P = _Pts[:,self.mask], _Pts_P[:,self.mask]

        

        rand_b_ix = list(range(b))
        random.shuffle(rand_b_ix)

        _Pts_rand = _Pts.clone()[rand_b_ix]
        
        self.grid_pts = _Pts_rand
    
        ImP_r, _Pts_P_r = self.A['ImP'][rand_ix], _Pts_P[rand_ix]
        Im_r, _Pts_r = self.A['Im'][rand_ix], _Pts[rand_ix]
        Im = self.A['Im']

        ImP_r, Im, Im_r = toGrey(ImP_r), toGrey(Im), toGrey(Im_r)

        X = self.warp.warp_img(Im, _Pts, self.grid_pts)
        X_r = self.rigid.warp_img(Im, _Pts, self.grid_pts)

        X_P = self.warp.warp_img(ImP_r, _Pts_P_r, self.grid_pts)
        X_P_r = self.warp.warp_img(Im_r, _Pts_r, self.grid_pts, reg=1e30)

        lambda_ = abs(1 - compute_area(X) / compute_area(X_r))
        
        l_r = lambda_
        l_c = (1 - l_r).abs()
        l_r = l_r

        if self.crop:
            if self.step > 100 == 0:
                X_P1, X = crop_grey(X_P.clone(), X, True)
                X_P2, X_r_1 = crop_grey(X_P.clone(), X_r.clone(), True)
                X_P_r, X_r_2 = crop_grey(X_P_r, X_r.clone(), True)

            else:   
                X_r_1 = X_r.clone()
                X_r_2 = X_r.clone()
                X_P1 = X_P.clone()
                X_P2= X_P.clone()
        else:
            X_P1 = X_P.clone()
            X_r_1 = X_r.clone()
            X_P2 = X_P.clone()
            X_r_2 = X_r.clone()
        
        
        loss = l_c * (1 - self.mssi(X_P1, X)) + l_r * (1 - self.mssi(X_P2, X_r_1))
        

        loss = loss.mean()
       
        
        perp_loss =  loss
        
        
                    
        self.loss['rec'] = perp_loss
        self.loss['perp'] = perp_loss

        self.loss['rec'].backward()
        
        if self.gradclip:            
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
            

        h_p = H[0][None,:,:].repeat(b,1,1,1)
        
        _P_PTS_n, _PTS_n = 4 * spatial_soft_argmax2d(H, False), 4 * spatial_soft_argmax2d(h_p, False)

        _X = self.A['Im'].detach()
        
        return {'Heatmap' : H, 
                'generated': X_P, 
                'Pts' : Pts,
                'Pts_P' : Pts_P,
                'genSamples': X,
                'Y_perm' : X_r
                }

