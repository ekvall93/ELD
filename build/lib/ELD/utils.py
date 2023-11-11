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
import random 
from sklearn.decomposition import PCA
from model import UniModal, MultiModal, Model3D

### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
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
### Reused code from ends

def preprocess(img):
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
    """Computes and stores the average and current value
    From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    """
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

def crop( image , size, tight=8):
        im = cv2.resize(image, dsize=(size,size),
                        interpolation=cv2.INTER_LINEAR)
        return im

def affine_trans(image,angle=None,size=None):
    """From: https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019, slightly modified"""
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
    dst = cv2.warpAffine(image, M, (nW,nH))
    
    return dst, M

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

class Reduce_IMG:
    def __init__(self):
        self.pca = None

    def reduce_img(self, image):
        #check if self.pca is not None
        if self.pca is None:
            #fit pca based on image and reduce to 3 dim
            self.pca = PCA(n_components=3).fit(image.reshape(-1, image.shape[-1]))    
        return self.pca.transform(image.reshape(-1, image.shape[-1])).reshape(image.shape[0], image.shape[1], 3)