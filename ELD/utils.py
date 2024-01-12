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
import requests
from typing import List
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,                   
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
import cv2 as cv

from ELD.model import toGrey

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

def predict_landmarks(fan: FAN, image: torch.Tensor, device='cuda')->torch.Tensor:
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
            pts = 4 * spatial_soft_argmax2d(fan(image.to(device)), False)
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

def create_target_images(image: torch.Tensor, target_index: int)->torch.Tensor:
    """create target images

    Args:
        images (torch.Tensor): images
        target_index (int): index of target image

    Returns:
        torch.Tensor: target images
    """
    dst_image = image[target_index].unsqueeze(0).repeat((image.size(0), 1, 1, 1))
    return dst_image

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
    


def predict_multimodal_landmarks(fan: FAN, image: torch.Tensor, ismod0:bool=True, device='cuda')->torch.Tensor:
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
            pts = 4 * spatial_soft_argmax2d(fan(image.to(device), make_mask(image, ismod0).to(device))[0], False)
    return pts

def download_images_urls(urlList: list)->list:
    """download images from urls

    Args:
        urlList (list): list of urls

    Returns:
        list: list of images
    """
    imageList = []
    for i, url in enumerate(urlList):
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        imageList.append(image)
    return imageList


square = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

def multi_dil(im: np.array, num: int, element: np.array=square)->np.array:
    """dilate image

    Args:
        im (np.array): image
        num (int): number of dilations
        element (np.array, optional): . filter. Defaults to square.

    Returns:
        np.array: dilated image
    """
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im: np.array, num: int, element: np.array=square)->np.array:
    """erode image

    Args:
        im (np.array): image
        num (int): number of erosions
        element (np.array, optional): filter. Defaults to square.

    Returns:
        np.array: eroded image
    """
    for i in range(num):
        im = erosion(im, element)
    return im


def cropBakground(painting: np.array, filter_bg:float=0.8)->Tuple[np.array, np.array, np.array]:
    """crop background from image

    Args:
        painting (np.array): image
        filter_bg (float, optional): filter. Defaults to 0.8.

    Returns:
        Tuple[np.array, np.array, np.array]: original image, cropped image, mask
    """
    gray_painting = rgb2gray(painting)
    binarized = gray_painting<filter_bg
    
    multi_dilated = multi_dil(binarized, 0)
    area_closed = area_closing(multi_dilated, 0)
    multi_eroded = multi_ero(area_closed, 0)
    opened = opening(multi_eroded)
    
    label_im = label(opened)
    regions = regionprops(label_im)
    
    painting_new = np.copy(painting)
    
    
    L = label_im[label_im!=0]
    counts = np.bincount(L.ravel())
    main_label = np.argmax(counts)
    mask = label_im != main_label
    painting_new[mask] = 255
    #painting_new[mask] = 0
    
    return painting, painting_new,mask

def white2black(img: np.array)->np.array:
    """set white pixels to black

    Args:
        img (np.array): image

    Returns:
        np.array: image
    """
    white_pixels = np.where(
    (img[:, :, 0] == 255) & 
    (img[:, :, 1] == 255) & 
    (img[:, :, 2] == 255)
)
    # set those pixels to white
    img[white_pixels] = [0,0,0]
    return img

def padImg(img:np.array, pad:int)->np.array:
    """pad image

    Args:
        img (np.array): image
        pad (int): padding

    Returns:
        np.array: padded image
    """
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
    
    return result

def downscale_images(imageList: List[np.array], scale_percent: int = 10)->List[np.array]:
    """Downscale images

    Args:
        imageList (List[np.array]): list of images
        scale_percent (int, optional): how much to downscale. Defaults to 10.

    Returns:
        List[np.array]: list of downscale images
    """
    for i, img in enumerate(imageList):
        
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        imageList[i] = img
    return imageList

def plot_images(imageList: List[np.ndarray])->None:
    """Plots a list of images in a grid

    Args:
        imageList (List[np.ndarray]): List of images to plot
    """
    # Calculate the number of rows and columns for the subplots
    num_images = len(imageList)
    num_rows = (num_images - 1) // 5 + 1
    num_cols = min(num_images, 5)

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten the axs array if it's more than 1 row
    if num_rows > 1:
        axs = axs.flatten()

    # Loop through the images and plot them
    for i, img in enumerate(imageList):
        ax = axs[i]
        ax.imshow(img)
        ax.axis('off')

    # Remove any empty subplots in the last row
    for i in range(num_images, num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    
def mask_background(imageList: List[np.ndarray], scale: float = 0.75)-> List[np.ndarray]:
    """

    Args:
        imageList (List[np.ndarray]): list of images
        scale (float, optional): How much to scale. Defaults to 0.75.

    Returns:
        List[np.ndarray]: list of images with background masked
    """
    for i, image in enumerate(imageList):
        painting, image, mask = cropBakground(image, scale)
        image = white2black(image)
        imageList[i] = image
    
    return imageList

def crop_non_tissue(imageList: List[np.array])->List[np.array]:
    """Crop non tissue from images

    Args:
        imageList (List[np.array]): list of images

    Returns:
        List[np.array]: list of cropped images
    """
    for i, image in enumerate(imageList):

        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        thresholded = cv.threshold(grayscale, 0, 255, cv.THRESH_OTSU)
        
        bbox = cv.boundingRect(thresholded[1])

        x, y, w, h = bbox
        
        image = image[y:y+h, x:x+w]
        
        imageList[i] = image
    return imageList

def downsize_and_save(imageList: List[np.array], path: str)->List[np.array]:
    """Downsize images and save them

    Args:
        imageList (List[np.array]): list of images
        path (str): path to save images

    Returns:
        List[np.array]: list of downsized images
    """
    small_imgs = list()
    for i, img in enumerate(imageList):
        
        small_img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
        small_imgs.append(small_img)
        small_img = padImg(small_img, 2)
    
        plt.imsave(f"{path}{i}.png", small_img)
    return small_imgs

def rescale_landmarks(pts: np.array, imageList: List[np.array])->List[np.array]:
    """_summary_

    Args:
        pts (np.array): landmarks
        imageList (List[np.array]): list of images

    Returns:
        List[np.array]: list of rescaled landmarks
    """
    scaled_pts = []

    for i, p in enumerate(pts):
        original_size = imageList[i].shape
        scale_x = original_size[0] / 128
        scale_y = original_size[1] / 128

        scaled_pts_current = p.clone()
        
        scaled_pts_current[:, 0] *= scale_y
        scaled_pts_current[:, 1] *= scale_x

        scaled_pts.append(scaled_pts_current)
    return scaled_pts


# Function to pad an image and adjust its corresponding landmarks
def _pad_image_and_adjust_landmarks(image: np.array, landmarks:np.array, square_size: int)->Tuple[np.array, np.array]:
    """Pad image and adjust landmarks

    Args:
        image (np.array): images
        landmarks (np.array): landmarks
        square_size (int): size of square

    Returns:
        Tuple[np.array, np.array]: padded image and adjusted landmarks
    """
    height, width = image.shape[:2]
    top_pad = (square_size - height) // 2
    bottom_pad = square_size - height - top_pad
    left_pad = (square_size - width) // 2
    right_pad = square_size - width - left_pad

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Adjust landmark coordinates
    adjusted_landmarks = landmarks.clone()  # Clone to avoid modifying the original
    adjusted_landmarks[:, 0] += left_pad  # Adjust x-coordinate
    adjusted_landmarks[:, 1] += top_pad   # Adjust y-coordinate

    return padded_image, adjusted_landmarks


def pad_image_and_adjust_landmarks(imageList: np.array, landmarks: np.array)->Tuple[np.array, np.array]:
    """Pad image and adjust landmarks

    Args:
        image (np.array): images
        landmarks (np.array):  landmarks
        square_size (int): size of square

    Returns:
        Tuple[np.array, np.array]: padded image and adjusted landmarks
    """
    # Assuming original_images is a list of your original images
    # And landmarks is a list of corresponding landmark tensors
    max_width = max(image.shape[1] for image in imageList)
    max_height = max(image.shape[0] for image in imageList)
    square_size = max(max_width, max_height)

    # Pad images and adjust landmarks
    padded_images = []
    adjusted_landmarks_list = []
    for img, lm in zip(imageList, landmarks):
        padded_img, adjusted_lm = _pad_image_and_adjust_landmarks(img, lm, square_size)
        padded_images.append(padded_img)
        adjusted_landmarks_list.append(adjusted_lm)
        
    padded_images_torch = torch.stack([preprocess(img) for img in padded_images])
    adjusted_landmarks = torch.stack(adjusted_landmarks_list)
    return padded_images_torch, adjusted_landmarks


def corr(batch1: torch.Tensor, batch2: torch.Tensor)->torch.Tensor:
    """get correlation between two batches

    Args:
        batch1 (torch.Tensor): batch 1
        batch2 (torch.Tensor): batch 2

    Returns:
        torch.Tensor: correlation
    """
    
    batch_size = batch1.size(0)
    
    batch1, batch2 = toGrey(batch1), toGrey(batch2)
    
    # Flatten the images
    batch1_flat = batch1.view(batch_size, -1)
    batch2_flat = batch2.view(batch_size, -1)

    # Normalize the data to have zero mean and unit variance
    batch1_flat = (batch1_flat - batch1_flat.mean(dim=1, keepdim=True)) / batch1_flat.std(dim=1, keepdim=True)
    batch2_flat = (batch2_flat - batch2_flat.mean(dim=1, keepdim=True)) / batch2_flat.std(dim=1, keepdim=True)

    # Compute the Pearson correlation
    pearson_correlation = (batch1_flat * batch2_flat).sum(dim=1) / (batch1_flat.size(1))
    return pearson_correlation

def plot_warped_images(mapped_imgs:torch.Tensor, mapped_pts: torch.Tensor, loss, square_size:int = 128, method: str = 'homography', device='cuda')->None:
    """plot warped images

    Args:
        mapped_imgs (torch.Tensor): Mapped images
        dst_pts (torch.Tensor): target landmarks
        loss (_type_): loss
        square_size (int, optional): size of landmarks. Defaults to 128.
        method (str, optional): method in title. Defaults to 'homography'.
    """
    np_img = toImg(mapped_imgs.to(device)[:,:3], mapped_pts, square_size)
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))  # adjust the size as needed
    axs = axs.ravel()

    mean_loss = round(np.mean(loss), 3)
    for i in range(11):
        img = np_img[i]
        axs[i].imshow(img)
        if i == 0:
            axs[i].set_title(f"Target")
        else: 
            l = loss[i - 1].item()
            l = round(l,3)
            axs[i].set_title(f"Warped: Image {i+1}\n Correlation: {l}")
        axs[i].axis('off')  # to hide the axis

    #add title to figure
    
    fig.suptitle(f"Overview of registered Images with {method}\n Average Correlation: {mean_loss:.3f}", fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.show()