from __future__ import print_function, division
import glob, os, sys, pickle, torch, cv2, time, numpy as np
from torch.utils.data import Dataset
from torchvision.utils import save_image
from shutil import copy2
import torch


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

# mystuff
#from model import model as HomoModel, modelElastic, modelHyper

from model import HomoModel


from databases import SuperDB
from utils import *
from Train_options import Options
import ipdb
import time

def getModel(model:str, hyper:bool):
    if model == "Homo":
        return HomoModel
    elif model == "SAIC_Homo_Patches" and not hyper:
        return modelSAIC_Homo_Patches
    elif model == "SAIC_Homo":
        return modelSAIC_Homo
    elif model == "Flow":
        return modelFlow
    else:
        raise ValueError("Model's available are: Homo, TPS, SAIC_TPS, and SAIC_TPS_Patches")

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



def main():
    # parse args
    global args, reducer
    args = Options().args
    reducer = Reduce_IMG()
    
    # copy all files from experiment
    cwd = os.getcwd()
    for ff in glob.glob("*.py"):
        copy2(os.path.join(cwd,ff), os.path.join(args.folder,'code'))

    # initialise seeds
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    np.random.seed(1000)

    # choose cuda
    if args.cuda == 'auto':
        import GPUtil as GPU
        GPUs = GPU.getGPUs()
        idx = [GPUs[j].memoryUsed for j in range(len(GPUs))]
        print(idx)
        assert min(idx) < 11.0, 'All {} GPUs are in use'.format(len(GPUs))
        idx = idx.index(min(idx))
        print('Assigning CUDA_VISIBLE_DEVICES={}'.format(idx))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # parameters
    sigma = float(args.s)
    temperature = float(args.t)
    gradclip = int(args.gc)
    npts = int(args.npts)
    bSize = int(args.bSize)
    angle = float(args.angle)
    flip = eval(str(args.flip))
    tight = int(args.tight)
    elastic = args.elastic
    hyper = bool(args.hyper)
    model = args.model
    elastic_sigma = float(args.elastic_sigma)
    ws = float(args.ws)
    warp = str(args.warp)

    if hyper:
        if not args.n_genes:
            raise RuntimeError("You need to provide n genes")
        else:
            n_genes = int(args.n_genes)
    else:
        n_genes = None

    
    model = getModel(model, hyper)

    video_dataset = SuperDB(path=args.data_path,sigma=sigma,size=args.size,flip=flip,angle=angle,tight=tight, elastic=elastic, bSize=bSize, hyper=hyper, elastic_sigma=elastic_sigma, ws=ws)
    n_chanels = video_dataset.get_num_channels()
    
    model = model(sigma=sigma,temperature=temperature, 
                          gradclip=gradclip, npts=npts, option=args.option, 
                          size=args.size, path_to_check=args.checkpoint, 
                          warmup_steps = ws, n_chanels = n_chanels, elastic=elastic, warp=warp)
    
    plotkeys = ['deformed_image1', 'deformed_image2','deformed_image1_rot','generated_rot','generated_deformed', 'rot_patches', 'deformed_patches', 'Im', 'ImP', 'generated', 'X_S', 'generated2', 'ImPD', 'genSamples']

    losskeys = list(model.loss.keys()) + ['lambda']
    
    # define plotters
    global plotter
    if not args.visdom:
        print('No Visdom')
        plotter = None
    else:
        from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomSaver, VisdomTextLogger
        experimentsName = str(args.visdom)
        plotter = dict.fromkeys(['images','losses'])
        plotter['images'] = dict( [ (key, VisdomLogger("images", port=int(args.port), env=experimentsName, opts={'title' : key})) for key in plotkeys ])
        plotter['losses'] = dict( [ (key, VisdomPlotLogger("line", port=int(args.port), env=experimentsName,opts={'title': key, 'xlabel' : 'Iteration', 'ylabel' : 'Loss'})) for key in losskeys]  )
        
    # prepare average meters
    global meters, l_iteration, loss, useGen
    meterskey = ['batch_time', 'data_time'] 
    meters = dict([(key,AverageMeter()) for key in meterskey])
    meters['losses'] = dict([(key,AverageMeter()) for key in losskeys])
    l_iteration = float(0.0)
    

    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.FAN.parameters())])
    print('FAN # trainable parameters: {}'.format(params))

   
    # define data
    #video_dataset = SuperDB(path=args.data_path,sigma=sigma,size=args.size,flip=flip,angle=angle,tight=tight, elastic=elastic, bSize=bSize, hyper=hyper, elastic_sigma=elastic_sigma, ws=ws)
    videoloader = FastDataLoader(video_dataset, batch_size=bSize, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    print('Number of workers is {:d}, and bSize is {:d}'.format(int(args.num_workers),bSize))
       
    # define optimizers
    lr_fan = args.lr_fan
    lr_gan = args.lr_gan
    lr_drift = args.lr_drift

    
    print('Using learning rate {} for FAN, and {} for GAN'.format(lr_fan,lr_gan))
    optimizerFAN = torch.optim.Adam(model.FAN.parameters(), lr=lr_fan , betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerFAN = torch.optim.lr_scheduler.StepLR(optimizerFAN, step_size=args.step_size, gamma=args.gamma)

    myoptimizers = {'FAN' : optimizerFAN}

    
    



    # path to save models and images
    path_to_model = os.path.join(args.folder,args.file)

    #start timing for epoch
    start = time.time()

    best_loss = float('inf')
    min_improvement = 1e-4
    patience_counter = 0
    patience = 5
    
    for epoch in range(0,160):
        #if model.train_fan:
        #    schedulerFAN.step()
        schedulerFAN.step()

        loss = train_epoch(videoloader, model, myoptimizers, epoch, bSize, elastic, args.size, hyper)
        model._save(path_to_model,epoch)

        
        # Check if early stopping should be performed

        improvement = best_loss - loss
        if improvement > min_improvement:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1


        #print epoc, loss and time in minutes with start time, patience_counter
        print('Epoch: [{0}]\t'
                'Loss {loss:.4f}\t'
                'Time {time:.2f}\t'
                'Patience {patience:.2f}\t'.format(epoch, loss=loss, time=(time.time() - start)/60, patience=patience_counter))

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    

def train_epoch(dataloader, model, myoptimizers, epoch, bSize, elastic, size, hyper):
    itervideo = iter(dataloader)
    global l_iteration, reducer
    
    log_epoch = {}
    end = time.time()
    #print(len(itervideo))
    #As many itertions as deaful SAIC projectgit s
    for i in range(0,300):
    
        # - get data
        all_data = next(itervideo,None) 
        if all_data is None:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)
        elif all_data['Im'].shape[0] < bSize:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)

        # - set batch
        
        model._set_batch(all_data)

        #time fowrd step with time
        
        # - forward
        output = model.forward(l_iteration)
        

       
            
        myoptimizers['FAN'].step()
                
        meters['losses']['rec'].update(model.loss['rec'].item(), bSize)
        meters['losses']['perp'].update(model.loss['rec'].item(), bSize)
        l_iteration = l_iteration + 1

        
        if i % 100 == 0:

            allimgs_deformed = None
            for (ii,imtmp) in enumerate(model.A['ImP'].to('cpu').detach()):
                hyper = imtmp.shape[0] >3
                if hyper:
                    #ipdb.set_trace()
                    #imtmp = imtmp[:3,:,:]
                    imtmp = imtmp.permute(1,2,0).numpy()
                    imtmp = reducer.reduce_img(imtmp)
                    improc = (255*imtmp.astype(np.uint8).copy())
                else:
                    improc = (255*imtmp.permute(1,2,0).numpy()).astype(np.uint8).copy()
                    

                x = output['Pts_P'][ii]
                for m in range(0,x.shape[0]):
                    if hyper:
                        cv2.circle(improc, (int(x[m,0]), int(x[m,1])), 2 * circle_size(size), (255, 255,255),-1)  
                    cv2.circle(improc, (int(x[m,0]), int(x[m,1])), circle_size(size), colors[m % 10],-1)

                if allimgs_deformed is None:
                    allimgs_deformed = np.expand_dims(improc,axis=0)
                else:
                    allimgs_deformed = np.concatenate((allimgs_deformed, np.expand_dims(improc,axis=0)))

            allimgs_image = None
            if 'Pts' in output.keys():
                for (ii,imtmp) in enumerate(model.A['Im'].to('cpu').detach()):
                    hyper = imtmp.shape[0] >3
                    if hyper:
                        imtmp = imtmp.permute(1,2,0).numpy().copy()
                        imtmp = reducer.reduce_img(imtmp)
                        improc = (255*imtmp.astype(np.uint8).copy())
                    else:
                        improc = (255*imtmp.permute(1,2,0).numpy()).astype(np.uint8).copy()

                    x = output['Pts'][ii]
                    for m in range(0,x.shape[0]):
                        if hyper:
                            cv2.circle(improc, (int(x[m,0]), int(x[m,1])), 2 * circle_size(size), (255, 255,255),-1)
                        
                        cv2.circle(improc, (int(x[m,0]), int(x[m,1])), circle_size(size), colors[m % 10],-1)

                    if allimgs_image is None:
                        allimgs_image = np.expand_dims(improc,axis=0)
                    else:
                        allimgs_image = np.concatenate((allimgs_image, np.expand_dims(improc,axis=0)))


            allimgs_imageP = None
            if 'Pts' in output.keys():
                for (ii,imtmp) in enumerate(model.A['ImPD'].to('cpu').detach()):
                    hyper = imtmp.shape[0] >3
                    if hyper:
                        imtmp = imtmp.permute(1,2,0).numpy().copy()
                        imtmp = reducer.reduce_img(imtmp)
                        improc = (255*imtmp.astype(np.uint8).copy())
                    else:
                        improc = (255*imtmp.permute(1,2,0).numpy()).astype(np.uint8).copy()

                    x = output['Pts'][ii]
                    for m in range(0,x.shape[0]):
                        cv2.circle(improc, (int(x[m,0]), int(x[m,1])), circle_size(size), colors[m % 10],-1)

                    if allimgs_imageP is None:
                        allimgs_imageP = np.expand_dims(improc,axis=0)
                    else:
                        allimgs_imageP = np.concatenate((allimgs_imageP, np.expand_dims(improc,axis=0)))



            if plotter is not None:
                
                
                plotter['images']['Im'].log(torch.from_numpy(allimgs_image).permute(0,3,1,2))
                plotter['images']['ImP'].log(torch.from_numpy(allimgs_deformed).permute(0,3,1,2))
                plotter['images']['ImPD'].log(torch.from_numpy(allimgs_imageP).permute(0,3,1,2))

                if output['generated'].cpu().data.shape[1] >3:
                    generated = None
                    for img in output['generated'].cpu().data:
                        img = img.permute(1,2,0).numpy()
                        img = reducer.reduce_img(img)
                        img = (255*img.astype(np.uint8).copy())
                        if generated is None:
                            generated = np.expand_dims(img,axis=0)
                        else:
                            generated = np.concatenate((generated, np.expand_dims(img,axis=0)))
                    plotter['images']['generated'].log(torch.from_numpy(generated).permute(0,3,1,2))
                else:
                    plotter['images']['generated'].log(output['generated'].cpu().data)


                if "genSamples" in output.keys():
                    if output['genSamples'].cpu().data.shape[1] >3:
                        generated = None
                        for img in output['genSamples'].cpu().data:
                            img = img.permute(1,2,0).numpy()
                            img = reducer.reduce_img(img)
                            img = (255*img.astype(np.uint8).copy())
                            if generated is None:
                                generated = np.expand_dims(img,axis=0)
                            else:
                                generated = np.concatenate((generated, np.expand_dims(img,axis=0)))
                        plotter['images']['genSamples'].log(torch.from_numpy(generated).permute(0,3,1,2))
                    else:
                        plotter['images']['genSamples'].log(output['genSamples'].cpu().data)

                                
                """ if model.GEN and model.ws_finnished:
                    if hyper:
                        generated2 = None
                        for img in output['generated2'].cpu().data:
                            img = img.permute(1,2,0).numpy()
                            img = reducer.reduce_img(img)
                            img = (255*img.astype(np.uint8).copy())
                            if generated2 is None:
                                generated2 = np.expand_dims(img,axis=0)
                            else:
                                generated2 = np.concatenate((generated2, np.expand_dims(img,axis=0)))
                        plotter['images']['generated2'].log(torch.from_numpy(generated2).permute(0,3,1,2))
                    else:
                        plotter['images']['generated2'].log(output['generated2'].cpu().data)

                    if hyper:
                        generated2 = None
                        for img in output['samples_X'].cpu().data:
                            img = img.permute(1,2,0).numpy()
                            img = reducer.reduce_img(img)
                            img = (255*img.astype(np.uint8).copy())
                            if generated2 is None:
                                generated2 = np.expand_dims(img,axis=0)
                            else:
                                generated2 = np.concatenate((generated2, np.expand_dims(img,axis=0)))
                        plotter['images']['genSamples'].log(torch.from_numpy(generated2).permute(0,3,1,2))
                    else:
                        plotter['images']['genSamples'].log(output['samples_X'].cpu().data)
                    
                    
                    #plotter['images']['genSamples'].log(output['samples_X'].cpu().data)

                        

                if model.GEN:
                    allimgs_image = None
                    for (ii,imtmp) in enumerate(all_data['ImPD'].to('cpu').detach()):
                        
                        if hyper:
                            #imtmp = imtmp[:3,:,:]
                            imtmp = imtmp.permute(1,2,0).numpy().copy()
                            imtmp = reducer.reduce_img(imtmp)
                            improc = (255*imtmp.astype(np.uint8).copy())
                        else:
                            improc = (255*imtmp.permute(1,2,0).numpy()).astype(np.uint8).copy()

                        
                        #x = 4*output['Points'][ii]
                        x = output['Pts_PD'][ii]

                        if torch.is_tensor(output['mapped_p']):
                            m_x = output['mapped_p'][ii]
                        
                        for m in range(0,x.shape[0]):
                            if hyper:
                                cv2.circle(improc, (int(x[m,0]), int(x[m,1])), 2 * circle_size(size), (255, 255,255),-1)
                            
                            cv2.circle(improc, (int(x[m,0]), int(x[m,1])), circle_size(size), colors[m % 10],-1)

                            if torch.is_tensor(output['mapped_p']):

                                cv2.circle(improc, (int(m_x[m,0]), int(m_x[m,1])), circle_size(size), colors[m % 10],2)

                        if allimgs_image is None:
                            allimgs_image = np.expand_dims(improc,axis=0)
                        else:
                            allimgs_image = np.concatenate((allimgs_image, np.expand_dims(improc,axis=0)))

                    plotter['images']['ImPD'].log(torch.from_numpy(allimgs_image).permute(0,3,1,2)) """



                #print(model.loss['rec'].item())
                #print(float(model.TPS.std.cpu()))
                #plotter['losses']['rec'].log( l_iteration, model.loss['rec'].item() )
                plotter['losses']['rec'].log( l_iteration, model.loss['perp'].item() )
                
                plotter['losses']['lambda'].log( l_iteration, model.loss['perp'].item() )


            
               
        log_epoch[i] = model.loss       
        meters['batch_time'].update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            mystr = 'Epoch [{}][{}/{}] '.format(epoch, i, len(dataloader))
            mystr += 'Time {:.2f} ({:.2f}) '.format(meters['data_time'].val , meters['data_time'].avg )
            mystr += ' '.join(['Loss: {:s} {:.3f} ({:.3f}) '.format(k, meters['losses'][k].val , meters['losses'][k].avg ) for k in meters['losses'].keys()])
            print( mystr )
            with open(args.folder + '/args_' + args.file[0:-8] + '.txt','a') as f: 
                print( mystr , file=f)

    with open(args.folder + '/args_' + args.file[0:-8] + '_' + str(epoch) + '.pkl','wb') as f:
        pickle.dump(log_epoch,f)  

    avg_loss =  meters['losses']['rec'].avg


    meters['losses']['rec'].reset()
    meters['losses']['perp'].reset()

    return avg_loss

if __name__ == '__main__':
    main()


