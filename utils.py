# The following implementation is based on the techniques described in:
# "Object Landmark Discovery Through Unsupervised Adaptation" by Enrique Sanchez and Georgios Tzimiropoulos
# You can find the article here: http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf
#
# For more details on the practical implementation of these techniques, check out the corresponding GitHub repository:
# https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019

import os, sys, time, torch, math, numpy as np, cv2, collections
#It will interfer with DataLoader otherwise
cv2.setNumThreads(0)
import torch.nn as nn
import torch.nn.functional as F
import elasticdeform
import numpy as np
import ipdb
import random 
import torchvision

from model import UniModal, MultiModal, Model3D

################################### - classes 

colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
          (255, 255, 255),
          (128, 128, 0),
          (128, 0, 128),
          (0, 128, 128)]
          
colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
          (255, 255, 255),
          (128, 128, 0),
          (128, 0, 128),
          (0, 128, 128)]

def preprocess(img):
    #img = img[1:129, 1:129,:]
    img = img/255.0
    
    img = torch.from_numpy(img.swapaxes(2,1).swapaxes(1,0))
    img = img.type_as(torch.FloatTensor())
    
    return img


def toImg(img, pts=None, size=128, set_pts=True):
    allimgs_deformed = None
    for (ii,imtmp) in enumerate(img.to('cpu').detach()):
        improc = (255*imtmp.permute(1,2,0).numpy()).astype(np.uint8).copy()
        
        
        if set_pts:
            x = pts[ii]
            for m in range(0,x.shape[0]):
                cv2.circle(improc, (int(x[m,0]), int(x[m,1])), circle_size(size), colors[m % 10],-1)

        if allimgs_deformed is None:
            allimgs_deformed = np.expand_dims(improc,axis=0)
        else:
            allimgs_deformed = np.concatenate((allimgs_deformed, np.expand_dims(improc,axis=0)))
    return allimgs_deformed

def circle_size(size):
    return int(3 * np.ceil(size / 128))
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class HeatMap(torch.nn.Module):
    """Defines a differentiable Gaussian heatmap"""
    def __init__(self, out_res, sigma=0.5):
        super(HeatMap, self).__init__()
        self.sigma = sigma
        print('The size of heatmap is {:f}'.format(self.sigma))
        y,x = torch.meshgrid([torch.arange(0,out_res).float(), torch.arange(0,out_res).float()])
        self.x = x
        self.y = y
        self.out_res = out_res
    
    def forward(self, pts):
        bSize, nPts = pts.size(0), pts.size(1)
        x = self.x.repeat(bSize,nPts,1,1)
        y = self.y.repeat(bSize,nPts,1,1)
        xscore = torch.unsqueeze(torch.unsqueeze(pts[:,:,0], 2),3)
        yscore = torch.unsqueeze(torch.unsqueeze(pts[:,:,1], 2),3)
        xscore = xscore - x.to(xscore.device)
        yscore = yscore - y.to(yscore.device)
        hms = -(xscore**2 + yscore**2)
        hms = torch.exp(hms/self.sigma)
        return hms
    
class SoftArgmax2D(torch.nn.Module):
    """ Implementation of a 2d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations."""
    def __init__(self, base_index=0, step_size=1, softmax_temp=1.0):
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp

    def _softmax_2d(self, x, temp):
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        smax = self._softmax_2d(x, self.softmax_temp)# * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(batch_size,channels,1,1)
        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).float().cuda()
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)
        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).float().cuda()
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)
        return torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)


class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.LossOutput = collections.namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        self.vgg_layers = vgg_model.features if hasattr(vgg_model,'features') else vgg_model.module.features #### to allow use in DataParallel
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.LossOutput(**output)


################################ functions
def process_image(image,angle=0, flip=False, sigma=1,size=128, tight=16, hmsize=64):
    output = dict.fromkeys(['image','M'])
    image = image/255.0
    
    if angle > 0:
        #tmp_angle = np.clip(np.random.randn(1) * angle, -40.0, 40.0)
        tmp_angle = np.clip(np.random.randn(1) * angle, -40.0, 40.0)
        image,M = affine_trans(image, tmp_angle)
        output['M'] = M
        #tight = int(tight + 4*np.random.randn())
    image = crop( image, size, tight )

    image = image.clip(0,1)
    image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
    output['image'] = image.type_as(torch.FloatTensor())

    
    

    return output

def padImg(img, pad):
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = old_image_width + pad
    new_image_height = old_image_height + pad
    #color = (255,255,255)
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
           x_center:x_center+old_image_width] = img
    
    result = cv2.resize(result, (130,130), interpolation = cv2.INTER_AREA)
    return result


def process_image_elastic(image,angle=0, flip=False, sigma=1,size=128, tight=16, hmsize=64, elastic_simga=5, rot_img=False):
    output = dict.fromkeys(['image','image_rot', 'image_deformed','points','M'])

    image = image/255.0

    if elastic_simga > 0.1:
        #get random number between 0 and 4
        _deformed_image = elasticdeform.deform_random_grid(image, axis=(0,1),sigma=elastic_simga * random.uniform(0,1), points=random.randint(3,9), cval=0, zoom=0.95, order = random.randint(0,5))
        _deformed_image_x = elasticdeform.deform_random_grid(image, axis=(0,1),sigma=elastic_simga * random.uniform(0,1), points=random.randint(3,9), cval=0, zoom=0.95, order = random.randint(0,5))
    else:
        _deformed_image = image.copy()
        _deformed_image_x = image.copy()

    #crop the region of interest


    n = _deformed_image_x.shape[-1]

    deformed_image = np.zeros((size + 4, size + 4, n))
    deformed_image[1:(size + 3), 1:(size + 3)] = _deformed_image_x.copy()

    deformed_image2 = np.zeros((size + 4, size + 4, n))
    deformed_image2[1:(size + 3), 1:(size + 3)] = _deformed_image.copy()

    if angle > 0:
        ### Image1
     
        max = 40
        tmp_angle = np.clip(np.random.randn(1) * angle, -max, max)
        deformed_image, M = affine_trans(deformed_image, tmp_angle)
        #output['M'] = M

        deformed_image2, _ = affine_trans(deformed_image2, tmp_angle)
        
        tmp_angle = np.clip(np.random.randn(1) * angle, -1 * max - tmp_angle, max - tmp_angle)
        deformed_image2, M = affine_trans(deformed_image2, tmp_angle)
        deformed_image3, M = affine_trans(deformed_image.copy(), tmp_angle)

        deformed_image3, M = affine_trans(deformed_image3, -1 * tmp_angle)


    if flip:
        image = cv2.flip(image, 1)

    deformed_image = deformed_image.clip(0,1)
    deformed_image = torch.from_numpy(deformed_image.swapaxes(2,1).swapaxes(1,0))
    output['image'] = deformed_image.type_as(torch.FloatTensor())

    deformed_image2 = deformed_image2.clip(0,1)
    deformed_image2 = torch.from_numpy(deformed_image2.swapaxes(2,1).swapaxes(1,0))
    output['image_rot'] = deformed_image2.type_as(torch.FloatTensor())

    deformed_image3 = deformed_image3.clip(0,1)
    deformed_image3 = torch.from_numpy(deformed_image3.swapaxes(2,1).swapaxes(1,0))
    output['image_deformed'] = deformed_image3.type_as(torch.FloatTensor())
    

    return output


def process_image_hyper(image,angle=0, flip=False,sigma=1,size=128, tight=16, hmsize=64):
    output = dict.fromkeys(['image','points','M'])
    
    splitedSize = 100

    h, w, c = image.shape
    a_splited = [image[:,:,x:x+splitedSize] for x in range(0, c, splitedSize)]

    r = np.random.randn(1) * angle
    t = np.random.randn()
    I_list = list()


    for I in a_splited:
        if angle > 0:
            tmp_angle = np.clip(r, -40.0, 40.0)
            I,M = affine_trans(I, tmp_angle)
            output['M'] = M
            tight = int(tight + 4*t)
        
        I = crop(I , size, tight)

        if flip:
            I = cv2.flip(I, 1)

        I = I/255.0
        I_list.append(I)

    
    image = np.concatenate(I_list, -1)
    
    image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
    output['image'] = image.type_as(torch.FloatTensor())
    
    
    return output


def crop( image , size, tight=8):
        im = cv2.resize(image, dsize=(size,size),
                        interpolation=cv2.INTER_LINEAR)
        return im

def affine_trans(image,angle=None,size=None):
    if angle is None:
        angle = 30*torch.randn(1)
       
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -float(angle), 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    #nW = int((h * sin) + (w * cos))
    #nH = int((h * cos) + (w * sin))
    nW = 128
    nH = 128
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    #dst = cv2.warpAffine(image, M, (nW,nH),borderMode=cv2.BORDER_REPLICATE)
    dst = cv2.warpAffine(image, M, (nW,nH))
    
    return dst, M



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

############################# - Visualisation utils

def savetorchimg(name,img):
    cv2.imwrite(name, cv2.cvtColor((255*img.permute(1,2,0).numpy()).astype(np.uint8) , cv2.COLOR_RGB2BGR))

def savetorchimgandpts(name,img,x,y=None):
    improc = (255*img.permute(1,2,0).numpy()).astype(np.uint8).copy()
    for m in range(0,x.shape[0]):
        cv2.circle(improc, (int(x[m,0]), int(x[m,1])), 2, (255,0,0),-1)
    if y is not None:
        for m in range(0,y.shape[0]):
            cv2.circle(improc, (int(y[m,0]), int(y[m,1])), 2, (0,255,0),-1)
    cv2.imwrite(name, cv2.cvtColor( improc , cv2.COLOR_RGB2BGR))


def saveheatmap(name,img):
    improc = cv2.applyColorMap( (255*img.permute(1,2,0).numpy()).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(name, cv2.cvtColor( improc , cv2.COLOR_RGB2BGR))


def savetorchimgandptsv2(name,img,x,thick=2,mSize=4): # to use different colours
    improc = (255*img.permute(1,2,0).numpy()).astype(np.uint8).copy()
    cv2.drawMarker(improc, (int(x[0,0]), int(x[0,1])), (255,0,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[1,0]), int(x[1,1])), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[2,0]), int(x[2,1])), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[3,0]), int(x[3,1])), (0,0,0), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[4,0]), int(x[4,1])), (255,255,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[5,0]), int(x[5,1])), (255,255,0), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[6,0]), int(x[6,1])), (255,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[7,0]), int(x[7,1])), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[8,0]), int(x[8,1])), (255,128,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[9,0]), int(x[9,1])), (0,0,128), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)       
    cv2.imwrite(name, cv2.cvtColor( improc , cv2.COLOR_RGB2BGR))


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def getModel(model:str):
    if model == "unimodal":
        return UniModal
    elif model == "multimodal":
        return MultiModal
    elif model == "3d":
        return Model3D
    else:
        raise ValueError("Model's available are: 'unimodal', and 'multimodal'")

#import pca
from sklearn.decomposition import PCA
class Reduce_IMG:
    def __init__(self):
        self.pca = None

    def reduce_img(self, image):
        #check if self.pca is not None
        if self.pca is None:
            #fit pca based on image and reduce to 3 dim
            self.pca = PCA(n_components=3).fit(image.reshape(-1, image.shape[-1]))    
        return self.pca.transform(image.reshape(-1, image.shape[-1])).reshape(image.shape[0], image.shape[1], 3)