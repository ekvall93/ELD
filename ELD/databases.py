# The following implementation is based on the techniques described in:
# "Object Landmark Discovery Through Unsupervised Adaptation" by Enrique Sanchez and Georgios Tzimiropoulos
# You can find the article here: http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf
#
# For more details on the practical implementation of these techniques, check out the corresponding GitHub repository:
# https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019

import inspect, torch, cv2, numpy as np
from torch.utils.data import Dataset
from .utils import process_image_elastic
from glob import glob
import tifffile as tiff
import re

x = torch.tensor([0])
x.share_memory_()

def get_frac(ws:int)->float:
    """Get the fraction of the warmup on multiprocesses

    Args:
        ws (int): warmup steps

    Returns:
        float: fraction of the warmup
    """
    global x
    x+=1
    val = min(x / ws, torch.tensor([1]))
    if x.item() % 1000 == 0:
        print(f"----- Warmup: {round(val.item(), 4) * 100}% ------")
    
    return val.item()

def extract_number_from_path(path):
    """Extract the number from the path

    Args:
        path (str): path to the image

    Returns:
        int: number of the image
    """
    match = re.search(r'(\d+)\.png$', path)
    if match:
        return int(match.group(1))
    return 0  # Default value if no match

class SuperDB(Dataset):
    def __init__(self, path=None, sigma=1, size=128, step=15, flip=False, angle=0, tight=16, nimages=2, affine=False, db='CelebA', bSize=32, elastic_sigma=5, ws=5_000, fileName=False, model="unimodal"):
        ### Inspiration from https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019 but is mostly everything is rewritten
        # - automatically add attributes to the class with the values given by the class
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        del args[0] 
        for k in args:
            setattr(self,k,values[k])
        
        self.model = model
        if self.model == "unimodal":
            self.preparedb_unimodal()
        elif self.model == "multimodal":
            self.preparedb_multimodal()
        elif self.model == "3d":
            self.preparedb_3D()

        self.db = db

    def _collect_unimodal(self,idx):
        path_to_img = self.files[idx]
        image = self.memory_bank[path_to_img]

        if self.fileName:
            return [image, path_to_img]
        else:
            return [image]
        
    def preparedb_unimodal(self):
       
        files = glob(f"{self.path}*")
        
        if files[0].endswith('.tif'):
            memory_bank = {f: tiff.imread(f) for f in files}
        else:
            memory_bank = {f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files}
        

        files = files * 10_000

        setattr(self, "files", files)
        setattr(self,'len', len(files))
        setattr(self,'memory_bank', memory_bank)
        #check if file is a tiff file
        
        if files[0].endswith('.tif'):
            setattr(self,'tiff',True)
        else:
            setattr(self,'tiff',False)

        setattr(self,'collect', self._collect_unimodal)
    
    def _collect_multimodal(self,idx):
        path_to_img = self.files[idx]
    
        image = self.memory_bank[path_to_img]

        mod = 0 if "mod1" in path_to_img else 1

        if self.fileName:
            return [image, mod, path_to_img]
        else:
            return [image, mod]
            
    def preparedb_multimodal(self):
        
        #Hard-coded for now....
        files = glob(f"{self.path}*")
        
        if files[0].endswith('.tif'):
            memory_bank = {f: tiff.imread(f) for f in files}
        else:
            memory_bank = {f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files}
        
        files = files * 10_000

        setattr(self, "files", files)
        setattr(self,'len', len(files))
        setattr(self,'memory_bank', memory_bank)

        if files[0].endswith('.tif'):
            setattr(self,'tiff',True)
        else:
            setattr(self,'tiff',False)
            
        setattr(self,'collect',self._collect_multimodal)

    def _collect_3d(self,idx):
        path_to_img = self.files[idx]
        image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
        if self.fileName:
            return [[image], path_to_img]
        else:
            return [[image], idx]
    # Define a function that returns the initialisation and the collect function
    def preparedb_3D(self):
       
        files = sorted(glob(f"{self.path}*"), key=extract_number_from_path)
        setattr(self, "files", files)
        setattr(self,'len', len(files))

        setattr(self,'collect',self._collect_3d)
       
    def get_num_channels(self):
        image = self.collect(0)
        if isinstance(image, list):
            image = image[0]
        return image[0].shape[-1]

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        if self.model == "3d":
            image, idx = self.collect(idx)
        else:
            image, idx = self.collect(idx), 0

        if self.model == "multimodal":
            image, mod = [image[0]], image[1]
        else:
            mod = 1

        nimg = len(image)

        frac = get_frac(self.ws)

        tmp_angle = self.angle * frac
        tmp_elastic_simga = self.elastic_sigma * frac

        if not self.affine:
            sample = dict.fromkeys(['Im', 'ImP', 'ImPD', 'mod', 'idx'], None)

            sample['idx'] = torch.Tensor([idx])

            flip = np.random.rand(1)[0] > 0.5 if self.flip else False # flip both or none
            
            for j in range(self.nimages):
                out = process_image_elastic(image=image[j%nimg],angle=(j+1)*tmp_angle, flip=flip, size=self.size, tight=self.tight,
                        elastic_simga=tmp_elastic_simga, rot_img=(j == 1))

                if j == 1:
                    sample['Im'] = out['image']
                    sample['mod'] = torch.Tensor([mod])
                    sample['ImP'] = out['image_rot']
                    sample['ImPD'] = out['image_deformed']
                
        else: 
            out = process_image_elastic(image=image[0],angle=0, flip=False, size=self.size, tight=self.tight, elastic_simga=0, rot_img=False)
            sample = {'Im': out['image'], 'fileName': image[1]}

        return sample


