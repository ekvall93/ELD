# The following implementation is based on the techniques described in:
# "Object Landmark Discovery Through Unsupervised Adaptation" by Enrique Sanchez and Georgios Tzimiropoulos
# You can find the article here: http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf
#
# For more details on the practical implementation of these techniques, check out the corresponding GitHub repository:
# https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019

import numpy as np, torch
from .MTFAN import FAN, MultiModalFAN, convertLayer
from .utils import *
from .warp import WARP, Rigid
from torch.autograd import Variable
from torchgeometry.contrib import spatial_soft_argmax2d
import random

from piqa import MS_SSIM
from typing import Tuple

def loadFan(npoints=10,n_channels=3,path_to_model=None, path_to_core=None)->FAN:
    """Loads the landmark detection model.
    From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    Args:
        npoints (int, optional): number of landmarks. Defaults to 10.
        n_channels (int, optional): number of channels. Defaults to 3.
        path_to_model (str, optional): path to model. Defaults to None.
        path_to_core (str, optional): path to core model. Defaults to None.

    Returns:
        FAN: Landmark detection model
    """
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

def loadMultiModalFan(npoints=10,n_channels=3,path_to_model=None, path_to_core=None)->MultiModalFAN:
    """Loads the landmark detection model.
    From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    Args:
        npoints (int, optional): number of landmarks. Defaults to 10.
        n_channels (int, optional): number of channels. Defaults to 3.
        path_to_model (str, optional): path to model. Defaults to None.
        path_to_core (str, optional): path to core model. Defaults to None.

    Returns:
        MultiModalFAN: Landmark detection model
    """
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


def crop(img1: torch.Tensor, img2: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    """Crop images based on each other

    Args:
        img1 (Tensor): Image 1
        img2 (Tensor): Image 2

    Returns:
        Tuple[img1, img2]: Masked image1 & image2
    """
    n = img1.size(1)
    G = img1.mean(1, keepdim=True)
    black_mask_Y_perm = (G < 0.1)
    img2 = torch.where(black_mask_Y_perm.repeat(1,n,1,1), torch.zeros_like(img2), img2)
    
    G = img2.mean(1, keepdim=True)
    black_mask = (G < 0.1)
    img1 = torch.where(black_mask.repeat(1,n,1,1), torch.zeros_like(img1), img1)

    return img1, img2


def toGrey(img: torch.Tensor)->torch.Tensor:
    """Convert to grayscale

    Args:
        img (Tensor): color image

    Returns:
        Tensor: grey image
    """
    return torch.einsum('bchw, c -> bhw', img, torch.tensor([0.299, 0.587, 0.114]).cuda()).unsqueeze(1)

def compute_area(img: torch.Tensor)->float:
    """Compute area of image that is not zero

    Args:
        img (Tensor): image

    Returns:
        float: number of pixels that are not zero
    """
    return (img.mean(1) > 0.1).flatten(1,2).sum(1)

def get_fraction_of_black(img1: torch.Tensor)->float:
    """Get fraction of black pixels

    Args:
        img1 (Tensor): image

    Returns:
        float: fraction of black pixels
    """
    G = img1
    black_mask_Y_perm = (G < 0.01)
    return black_mask_Y_perm.squeeze().flatten(1,2).sum(1) / (img1.shape[2] * img1.shape[3])


class UniModal():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10, option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, n_chanels = 3, warp='tps', crop=True):
        self.n_chanels = n_chanels
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        self.npoints = npts
        self.gradclip = gradclip
    
        self.FAN = FAN(1,n_points=self.npoints, in_channels=self.n_chanels)
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
                
        # - multiple GPUs
        if torch.cuda.device_count() > 1:
            self.FAN = torch.nn.DataParallel(self.FAN)
              
        self.FAN.to('cuda').train()
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None
        ### Reused code ends
        
        
        self.warp = WARP(warp)
        self.mssi = MS_SSIM(window_size=5,padding=True, n_channels=self.n_chanels).cuda()
        self.mssi_grey = MS_SSIM(window_size=5,padding=True, n_channels=1).cuda()
        self.mask = np.ones(self.npoints).astype(bool)
        self.crop = crop
        self.step = 0
        

    def _resume(self,path_fan: str)->None:
        """load FAN model
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            path_fan (str): path to model
        """
        self.FAN.load_state_dict(torch.load(path_fan))

    def _save(self, path_to_models: str, epoch: int)->None:
        """Save model
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            path_to_models (str): path to model
            epoch (int): epoch
        """
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
        
    def _set_batch(self,data: dict)->None:
        """Set batch to device
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            data (dict): data dict
        """
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}

    def new_mask(self)->None:
        """
            create new mask
        """
        self.mask = np.random.choice(2, self.npoints, p=[0.1, 0.9]).astype(bool)

    def set_mask(self, p=0.5)->None:
        """Set mask

        Args:
            p (float, optional): mask probability. Defaults to 0.5.
        """
        if random.uniform(0,1)>p:
            self.new_mask()
            if random.uniform(0,1)>p:
                self.new_mask()
            else:
                self.mask = np.ones(self.npoints).astype(bool)

    def forward(self, n_iter):
        b = self.A['Im'].shape[0]
        self.step   += 1
        
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        self.FAN.zero_grad()
        X = torch.cat([self.A['Im'], self.A['ImP']],0)
        H_X = self.FAN(X)
        H, H_P = torch.split(H_X, [b, b], dim=0)
        
        Pts = 4 * spatial_soft_argmax2d(H, False)
        Pts_P = 4 * spatial_soft_argmax2d(H_P, False)
        ### Reused code ends
        
        rand_ix = list(range(b))
        random.shuffle(rand_ix)
        
        _Pts, _Pts_P = Pts, Pts_P
        self.set_mask()
        #_Pts, _Pts_P = Pts[:,self.mask], Pts_P[:,self.mask]
        #_Pts, _Pts_P = Pts[:,self.mask], Pts_P[:,self.mask]
  
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
        
        perp_loss =  R_loss + 0.3 * consistent_loss

        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        self.loss['rec'] = perp_loss
        self.loss['perp'] = perp_loss
        self.loss['rec'].backward()
        
        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
        ### Reused code ends
        
        return {'Heatmap' : H_P, 
                'generated': X, 
                'Pts' : Pts,
                'Pts_P' : Pts_P,
                'genSamples': Y
                }

class MultiModal():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10,                option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, n_chanels = 3, warp='tps', crop=True):
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        self.npoints = npts
        self.gradclip = gradclip
        self.FAN = MultiModalFAN(1,n_points=self.npoints)
        
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
                
        # - multiple GPUs
        if torch.cuda.device_count() > 1:
            self.FAN = torch.nn.DataParallel(self.FAN)
            
        self.FAN.to('cuda').train()    
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None
        
        # - define losses for reconstruction
        self.SelfLoss = torch.nn.MSELoss().to('cuda')
        ### Reused code ends
        
        self.n_chanels = n_chanels
        self.warp = WARP(warp)
        self.mssi = MS_SSIM(window_size=5,padding=True, n_channels=self.n_chanels).cuda()
        self.step = 0
        self.new_mask()
        self.crop = crop

                                                                                                   
    def _resume(self,path_fan: str)->None:
        """load FAN model
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            path_fan (str): path to model
        """
        self.FAN.load_state_dict(torch.load(path_fan))

    def _save(self, path_to_models: str, epoch: int)->None:
        """Save model
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            path_to_models (str): path to model
            epoch (int): epoch
        """
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
        
    def _set_batch(self,data: dict)->None:
        """Set batch to device
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            data (dict): data dict
        """
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}

    def const(self,anneal_start=0, anneal_end=1_000):
        """Compute constant for warmup

        Args:
            anneal_start (int, optional): start of annealing. Defaults to 0.
            anneal_end (_type_, optional): end of annealing. Defaults to 1_000.

        Returns:
            float: warmup constant
        """
        return np.clip((self.step - anneal_start)/(anneal_end - anneal_start), 0.0, 1.0)

    def new_mask(self):
        """Create new mask
        """
        self.mask = np.random.choice(2, self.npoints, p=[0.2, 0.8]).astype(bool)
        if np.sum(self.mask) == self.npoints:
                self.mask[np.random.randint(0,self.npoints)] = False

    def set_mask(self, p=0.5):
        """Set mask

        Args:
            p (float, optional): Probility to mask. Defaults to 0.5.
        """
        if random.uniform(0,1)>p:
            self.new_mask()
            if random.uniform(0,1)>p:
                self.new_mask()
            else:
                self.mask = np.ones(self.npoints).astype(bool)
                self.mask[np.random.randint(0,self.npoints)] = False

    def forward(self, n_iter):
        self.A['mod'] = self.A['mod'].squeeze()
        self.step +=1 
        self.FAN.zero_grad()
        
        b = self.A['Im'].shape[0]
        rand_ix = list(range(b))
        random.shuffle(rand_ix)

        X = torch.cat([self.A['Im'], self.A['ImP']],0)
        MOD = torch.cat([self.A['mod'], self.A['mod']],0)
        
        H_X, A =  self.FAN(X, MOD == 1)
        
        #Get Im's activations
        A = [a[:b] for a in A]
        
        H, H_P = torch.split(H_X, [b, b], dim=0)

        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Pts = 4 * spatial_soft_argmax2d(H, False)
        Pts_P = 4 * spatial_soft_argmax2d(H_P, False)
        ### Reused code ends

        self.set_mask()

        _Pts, _Pts_P = Pts, Pts_P


        _Pts, _Pts_P = _Pts[:,self.mask], _Pts_P[:,self.mask]

        Y = self.warp.warp_img(self.A['ImP'], _Pts_P, _Pts)  

        """ if self.crop:     
            Y, Im = crop(Y, self.A['Im'].clone())
        else:
            Y, Im = Y, self.A['Im'].clone() """

        Y, Im = Y, self.A['Im'].clone()
        R_loss =  (1 - self.mssi(Y, Im))

       
        Z, mod = self.warp.warp_img(self.A['ImP'][rand_ix], _Pts_P[rand_ix], _Pts), self.A['mod'][rand_ix]
        H_Z, A_Z = self.FAN(Z, mod == 1)
        
        
        
        
        #ipdb.set_trace()
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
        random.shuffle(rand_ix_m1)
        
        rand_ix_m2 = list(range(len(X_mod2)))
        random.shuffle(rand_ix_m2)

        X_mod1_ = self.warp.warp_img(X_mod1[rand_ix_m1], _Pts_P_mod1[rand_ix_m1], _Pts_P_mod1)
        X_mod2_ = self.warp.warp_img(X_mod2[rand_ix_m2], _Pts_P_mod2[rand_ix_m2], _Pts_P_mod2)
        
        if self.crop:
            X_mod1, X_mod1_ = crop(X_mod1, X_mod1_)
            X_mod2_, X_mod2 = crop(X_mod2_, X_mod2)

        inter_mod_loss = 0.5 * ((1 - self.mssi(X_mod1, X_mod1_)) + (1 - self.mssi(X_mod2_, X_mod2)))

        #perp_loss = R_loss +   0.1 * self.const(anneal_end=500) * l + 0.1 * inter_mod_loss
        perp_loss = R_loss +   10 * self.const(anneal_end=500) * l + 0.1 * inter_mod_loss
        
        if self.step % 10 ==0:
            print('R_loss: ', R_loss.item(), 'inter_mod_loss: ', inter_mod_loss.item(), 'mod_loss', 0.1 * self.const(anneal_end=500) * l )
       

        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        self.loss['rec'] = perp_loss
        self.loss['perp'] = perp_loss
        self.loss['rec'].backward()
        
        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
        ### Reused code ends
            
        return {'Heatmap' : H_P, 
                'generated': Y, 
                'Pts' : Pts,
                'Pts_P' : Pts_P,
                'genSamples': H_Z
                }

class Model3D():
    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10,                option='incremental', size=128, path_to_check='checkpoint_fansoft/fan_109.pth',warmup_steps = 10_000, hyper=False, n_genes = None, warp='tps', n_chanels=3, crop=True):
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        self.npoints = npts
        self.gradclip = gradclip
        self.FAN = FAN(1,n_points=self.npoints)
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

        self.FAN.to('cuda').train()
        
        self.loss = dict.fromkeys(['rec', 'perp'])
        self.A = None
        self.SelfLoss = torch.nn.MSELoss().to('cuda')
        ### Reused code ends
        
        self.warmup_steps = warmup_steps
        self.n_genes = n_genes
        self.train_fan = True
        self.train_tps = True

        self.samples = 0
        
        self.warmpup_finnished = False
        self.hyper = hyper
        self.n_sucess = 0
        self.size = size
      
        self.TPS = None
        self.warp = WARP(warp)
        
        self.mssi = MS_SSIM(window_size=5, reduction='none', n_channels=1).cuda()
        self.step = 0

        self.new_mask()
        self.rigid = Rigid()

        self.crop = crop


    def _resume(self,path_fan: str)->None:
        """load FAN model
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            path_fan (str): path to model
        """
        self.FAN.load_state_dict(torch.load(path_fan))

    def _save(self, path_to_models: str, epoch: int)->None:
        """Save model
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            path_to_models (str): path to model
            epoch (int): epoch
        """
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
        
    def _set_batch(self,data: dict)->None:
        """Set batch to device
        From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        Args:
            data (dict): data dict
        """
        self.A = {k: Variable(data[k],requires_grad=True).to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}
        
    def const(self,anneal_start=0, anneal_end=1_000)->float:
        """Compute constant for warmup"""
        return np.clip((self.step - anneal_start)/(anneal_end - anneal_start), 0.0, 1.0)

    def new_mask(self):
        """Create new mask
        """
        self.mask = np.random.choice(2, self.npoints, p=[0.2, 0.8]).astype(bool)
        if np.sum(self.mask) == self.npoints:
                self.mask[np.random.randint(0,self.npoints)] = False

    def set_mask(self, p=0.5):
        """Set mask

        Args:
            p (float, optional): Probility to mask. Defaults to 0.5.
        """
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
        
        ### Reused code from https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019 begins
        self.FAN.zero_grad()
        X = torch.cat([self.A['Im'], self.A['ImP']],0)
        H_X = self.FAN(X)
        H, H_P = torch.split(H_X, [b, b], dim=0)
        
        Pts = 4 * spatial_soft_argmax2d(H, False)
        Pts_P = 4 * spatial_soft_argmax2d(H_P, False)
        ### Reused code ends
        
        
        rand_ix = list(range(b))
        random.shuffle(rand_ix)
        self.samples += b

        step = 4
        rand_ix = [random.randint(max(0, i-step), min(b-1, i+step)) for i in range(b)]
        step = 50
    
        self.set_mask()
        _Pts, _Pts_P = Pts, Pts_P
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

        lambda_ = abs(1 - compute_area(X) / compute_area(X_r))
        
        l_r = lambda_
        l_c = (1 - l_r).abs()

       
        X_P1 = X_P.clone()
        X_r_1 = X_r.clone()
        X_P2 = X_P.clone()
        
        
        loss = l_c * (1 - self.mssi(X_P1, X)) + l_r * (1 - self.mssi(X_P2, X_r_1))
        perp_loss =  loss.mean()
        
        
        ### Reused code from https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019 begins   
        self.loss['rec'] = perp_loss
        self.loss['perp'] = perp_loss
        self.loss['rec'].backward()
        
        if self.gradclip:            
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
        ### Reused code ends
            
        return {'Heatmap' : H, 
                'generated': X_P, 
                'Pts' : Pts,
                'Pts_P' : Pts_P,
                'genSamples': X,
                'Y_perm' : X_r
                }

