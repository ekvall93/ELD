import inspect, torch, pickle, cv2, os, numpy as np, scipy.io as sio
from torch.utils.data import Dataset
from utils import process_image, crop, affine_trans, process_image_elastic
#process_image_hyper
#process_image_hyper_elastic, process_image_hyper
from glob import glob
import time
import ipdb
import tifffile as tiff


x = torch.tensor([0])
x.share_memory_()

def get_frac(ws):
    global x
    x+=1
    val = min(x / ws, torch.tensor([1]))
    if x.item() % 1000 == 0:
        print(f"----- Warmup: {round(val.item(), 4) * 100}% ------")
    
    return val.item()

class SuperDB(Dataset):

    def __init__(self, path=None, sigma=1, size=128, step=15, flip=False, angle=0, tight=16, nimages=2, affine=False, db='CelebA', elastic=False, bSize=32, hyper=False,
    elastic_sigma=5, ws=5_000, fileName=False):
        # - automatically add attributes to the class with the values given by the class
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        del args[0] 
        for k in args:
            setattr(self,k,values[k])
        preparedb(self,hyper, bSize)
        self.db = db

    def get_num_channels(self):
        image = self.collect(self,0)
        return image[0].shape[-1]

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        image = self.collect(self,idx)
        nimg = len(image)

        frac = get_frac(self.ws)

        tmp_angle = self.angle * frac
        tmp_elastic_simga = self.elastic_sigma * frac

        if not self.affine:
            if self.elastic:
                sample = dict.fromkeys(['Im', 'ImP', 'ImPD'], None)
            else:
                sample = dict.fromkeys(['Im', 'ImP'], None)

            flip = np.random.rand(1)[0] > 0.5 if self.flip else False # flip both or none
            
            for j in range(self.nimages):
                if self.elastic:
                    if not self.hyper:
                        out = process_image_elastic(image=image[j%nimg],angle=(j+1)*tmp_angle, flip=flip, size=self.size, tight=self.tight,
                        elastic_simga=tmp_elastic_simga, rot_img=(j == 1))
                    else:
                        out = process_image_hyper_elastic(image=image[j%nimg],angle=(j+1)*tmp_angle, flip=flip, size=self.size, tight=self.tight,
                        elastic_simga=tmp_elastic_simga)
                        
                        
                else:
                    if self.hyper:
                        out = process_image_hyper(image=image[j%nimg],angle=(j+1)*tmp_angle, flip=flip, size=self.size, tight=self.tight) 
                    else:
                        out = process_image(image=image[j%nimg],angle=(j+1)*tmp_angle, flip=flip, size=self.size, tight=self.tight)
                if j == 1:
                    sample['Im'] = out['image']
                    #sample['Im_n'] = out['image_n']
                    if self.elastic:
                        sample['ImP'] = out['image_rot']
                        #sample['ImP_n'] = out['image_rot_n']
                        sample['ImPD'] = out['image_deformed']
                        #sample['ImPD_n'] = out['image_deformed_n']
                else:
                    if not self.elastic:
                        sample['ImP'] = out['image'] 
                        #sample['ImP_n'] = out['image_n'] 
            
           
        else: 
            """ image,M = affine_trans(image[0], 0)
            image = crop( image, 128, self.tight )
            image = image/255.0
            image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
            image = image.type_as(torch.FloatTensor()) """
            #print(image)
            out = process_image_elastic(image=image[0],angle=0, flip=False, size=self.size, tight=self.tight, elastic_simga=0, rot_img=False)
            sample = {'Im': out['image'], 'fileName': image[1]}

            """ tmp_angle = np.clip(np.random.randn(1) * self.angle, -40.0, 40.0) if self.angle > 0 else self.angle
            image,points,_ = affine_trans(image[0],points[0], tmp_angle)
            image, points = crop(image, points, 128, tight=self.tight)
            tmp_angle = np.random.randn(1) * self.angle
            imrot,ptsrot,M = affine_trans(image,points, size=128, angle=tmp_angle)
            image = image/255.0
            image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
            image = image.type_as(torch.FloatTensor())
            imrot = torch.from_numpy(imrot/255.0).permute(2,0,1).type_as(torch.FloatTensor())
            sample = {'Im': image, 'ImP': imrot, 'M' : torch.from_numpy(M).type_as(torch.FloatTensor()) , 'pts': points} """



        return sample


# Define a function that returns the initialisation and the collect function
def preparedb(self, hyper, bSize):
    if not hyper:
        def init(self):
            #Hard-coded for now....
            files = glob(f"{self.path}*")
            #while len(files) < bSize:
            #    files += files

            #load files to memory and save them in dict of filenames and images
            #check of tif
            if files[0].endswith('.tif'):
                memory_bank = {f: tiff.imread(f) for f in files}
            else:
                memory_bank = {f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files}
                
            

            #make 10_000 copies of the files


        
            




            files = files * 10_000

            setattr(self, "files", files)
            setattr(self,'len', len(files))
            setattr(self,'memory_bank', memory_bank)
            #check if file is a tiff file
            
            if files[0].endswith('.tif'):
                setattr(self,'tiff',True)
            else:
                setattr(self,'tiff',False)

        def collect(self,idx):
            path_to_img = self.files[idx]
            
            """ if self.tiff:
                image = tiff.imread(path_to_img)
            else:
                image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB) """

            image = self.memory_bank[path_to_img]

            #ipdb.set_trace()
            
            if self.fileName:
                return [image, path_to_img]
            else:
                return [image]

        init(self)
        setattr(self,'collect',collect)
        

    if hyper:
        def init(self):
            #Hard-coded for now....
            files = glob(f"{self.path}*")
            while len(files) < bSize:
                files += files
            setattr(self, "files", files)
            setattr(self,'len', len(files))

        def collect(self,idx):
            path_to_img = self.files[idx]
            image = np.load(path_to_img)
            if self.fileName:
                return [image, path_to_img]
            else:
                return [image]
        init(self)
        setattr(self,'collect',collect)
