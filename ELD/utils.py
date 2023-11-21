# The following implementation is based on the techniques described in:
# "Object Landmark Discovery Through Unsupervised Adaptation" by Enrique Sanchez and Georgios Tzimiropoulos
# You can find the article here: http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf
#
# For more details on the practical implementation of these techniques, check out the corresponding GitHub repository:
# https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019

import torch, cv2
#It will interfer with DataLoader otherwise
cv2.setNumThreads(0)
import torch.nn.functional as F
import elasticdeform
import numpy as np
import random 
from sklearn.decomposition import PCA
from .model import UniModal, MultiModal, Model3D
from glob import glob
import tifffile as tiff
from torchgeometry.contrib import spatial_soft_argmax2d
from .MTFAN import FAN
from typing import Tuple

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

def preprocess(img: np.array)->torch.Tensor:
    """preprocess image

    Args:
        img (np.array): image

    Returns:
        torch.Tensor: preprocessed image
    """
    img = img/255.0
    img = torch.from_numpy(img.swapaxes(2,1).swapaxes(1,0))
    img = img.type_as(torch.FloatTensor())
    return img


def toImg(img:torch.Tensor, pts:torch.Tensor=None, size:int=128, set_pts:bool=True)->np.array:
    """convert image to numpy array

    Args:
        img (torch.Tensor): image
        pts (torch.Tensor, optional): landmarks. Defaults to None.
        size (int, optional): image size. Defaults to 128.
        set_pts (bool, optional): set landmarks to image. Defaults to True.

    Returns:
        np.array: image as numpy array
    """
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

def circle_size(size: int)->int:
    """get circle size

    Args:
        size (int): image size

    Returns:
        int: image size
    """
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

    def update(self, val: int, n: int=1)->None:
        """update average meter

        Args:
            val (int): value
            n (int, optional): multiplier. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def process_image_elastic(image: np.array,angle:float=0, flip: bool=False, sigma: float=1,size:int=128, tight:int=16, hmsize:int=64, elastic_simga: float=5, rot_img:bool=False)->torch.Tensor:
    """process image

    Args:
        image (np.array): image
        angle (float, optional): _description_. Defaults to 0.
        flip (bool, optional): _description_. Defaults to False.
        sigma (float, optional): _description_. Defaults to 1.
        size (int, optional): _description_. Defaults to 128.
        tight (int, optional): _description_. Defaults to 16.
        hmsize (int, optional): _description_. Defaults to 64.
        elastic_simga (float, optional): _description_. Defaults to 5.
        rot_img (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
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
    
    
def load_imgs(path: str, sort:bool= False)->list:
    """load images from path

    Args:
        path (str): path to image folder

    Returns:
        list: list of images
    """
    #get files with glob
    files = glob(f"{path}*")
    
    if sort:
        #sort files by number
        files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    #load images
    if files[0].endswith('.tif'):
        imgs = [tiff.imread(f) for f in files]
    else:
        imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    return imgs

def predict_landmarks(fan: FAN, image: torch.Tensor)->torch.Tensor:
    """Predict landmarks from image

    Args:
        fan (FAN): trained landmark detector
        image (torch.Tensor): image to predict landmarks from

    Returns:
        torch.Tensor: landmarks
    """
    with torch.no_grad():
        fan.eval()
        with torch.no_grad():
            pts = 4 * spatial_soft_argmax2d(fan(image.cuda()), False)
    return pts

def create_target_landmarks(pts: torch.Tensor, target_index: int)->torch.Tensor:
    """create target landmarks

    Args:
        pts (torch.Tensor): landmarks
        target_index (int): index of target image

    Returns:
        torch.Tensor: target landmarks
    """
    dst_pts = pts[target_index].unsqueeze(0).repeat((pts.size(0), 1, 1))
    return dst_pts

def load_multi_modal_imgs(path:str)->Tuple[list, list]:
    """load multimodal images

    Args:
        path (str): path to image folder

    Returns:
        Tuple[list, list]: tuple of images
    """
    #get files with glob
    files = glob(f"{path}*")
    #sort files by number
    

    mod0, mod1 = [], []
    for f in files:
        if "mod0" in f:
            mod0.append(f)
        elif "mod1" in f:
            mod1.append(f)
    
    #load images
    if mod0[0].endswith('.tif'):
        mod0 = [tiff.imread(f) for f in mod0]
    else:
        mod0 = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in mod0]

    #load images
    if mod1[0].endswith('.tif'):
        mod1 = [tiff.imread(f) for f in mod1]
    else:
        mod1 = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in mod1]


    return mod0, mod1

def make_mask(img: torch.Tensor, is_true:bool=True)->torch.Tensor:
    """make mask for multimodal images

    Args:
        img (torch.Tensor): image
        is_true (bool, optional): If it's first modality. Defaults to True.

    Returns:
        torch.Tensor: mask
    """
    if is_true:
        return torch.ones(img.shape[0], dtype=torch.bool)
    else:
        return torch.zeros(img.shape[0], dtype=torch.bool)
    


def predict_multimodal_landmarks(fan: FAN, image: torch.Tensor, ismod0:bool=True)->torch.Tensor:
    """Predict landmarks for multimodal images

    Args:
        fan (FAN): Landmark model
        image (torch.Tensor): images to predict landmarks from
        ismod0 (bool, optional): if it's the first modality. Defaults to True.

    Returns:
        torch.Tensor: landmarks
    """
    with torch.no_grad():
        fan.eval()
        with torch.no_grad():
            pts = 4 * spatial_soft_argmax2d(fan(image.cuda(), make_mask(image, ismod0).cuda())[0], False)
    return pts