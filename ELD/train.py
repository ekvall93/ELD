# The following implementation is based on the techniques described in:
# "Object Landmark Discovery Through Unsupervised Adaptation" by Enrique Sanchez and Georgios Tzimiropoulos
# You can find the article here: http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf
#
# For more details on the practical implementation of these techniques, check out the corresponding GitHub repository:
# https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019

from __future__ import print_function, division
import glob, os, sys, pickle, torch, cv2, time, numpy as np
from shutil import copy2
import torch
from .databases import SuperDB
from .utils import * 
from .Train_options import Options
import time
import glob


def main():
    global reducer
    reducer = Reduce_IMG()
    
    ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    # parse args
    global args
    args = Options().args
    
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
    crop = bool(args.crop)
    
    ### Reused code from ends
    
    
    model_type = args.model
    elastic_sigma = float(args.elastic_sigma)
    ws = float(args.ws)
    warp = str(args.warp)
    patience = int(args.patience)

    
    model = getModel(model_type)

    video_dataset = SuperDB(path=args.data_path,sigma=sigma,size=args.size,flip=flip,angle=angle,tight=tight, bSize=bSize, elastic_sigma=elastic_sigma, ws=ws, model=model_type)
    n_chanels = video_dataset.get_num_channels()
    
    model = model(sigma=sigma,temperature=temperature, 
                          gradclip=gradclip, npts=npts, option=args.option, 
                          size=args.size, path_to_check=args.checkpoint, 
                          warmup_steps = ws, n_chanels = n_chanels,  warp=warp, crop=crop)
    
    plotkeys = ['deformed_image1', 'deformed_image2','deformed_image1_rot','generated_rot','generated_deformed', 'rot_patches', 'deformed_patches', 'Im', 'ImP', 'generated', 'X_S', 'generated2', 'ImPD', 'genSamples']

    ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    losskeys = list(model.loss.keys())
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
    ### Reused code from ends

    
    # define data
    if model_type == '3d':
        n_imgs = len(glob.glob(f"{args.data_path}/*"))
        videoloader = FastDataLoader(video_dataset, batch_size=n_imgs, shuffle=False, num_workers=int(args.num_workers), pin_memory=True)
    else:
        videoloader = FastDataLoader(video_dataset, batch_size=bSize, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)

    
    print('Number of workers is {:d}, and bSize is {:d}'.format(int(args.num_workers),bSize))
    
    ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    # define optimizers
    lr_fan = args.lr_fan
    print('Using learning rate {} for FAN'.format(lr_fan))
    optimizerFAN = torch.optim.Adam(model.FAN.parameters(), lr=lr_fan , betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerFAN = torch.optim.lr_scheduler.StepLR(optimizerFAN, step_size=args.step_size, gamma=args.gamma)
    
    ### Reused code from ends

    myoptimizers = {'FAN' : optimizerFAN}

    
    # path to save models and images
    path_to_model = os.path.join(args.folder,args.file)

    #start timing for epoch
    start = time.time()

    best_loss = float('inf')
    min_improvement = 1e-4
    patience_counter = 0
    
    
    for epoch in range(0,160):
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
        schedulerFAN.step()
        loss = train_epoch(videoloader, model, myoptimizers, epoch, bSize, args.size)
        model._save(path_to_model,epoch)
        ### Reused code from ends
 
        # Check if early stopping should be performed
        improvement = best_loss - loss
        if improvement > min_improvement:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        print('Epoch: [{0}]\t'
                'Loss {loss:.4f}\t'
                'Time {time:.2f}\t'
                'Patience {patience:.2f}\t'.format(epoch, loss=loss, time=(time.time() - start)/60, patience=patience_counter))

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

def train_epoch(dataloader, model, myoptimizers, epoch, bSize, size):
    ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    itervideo = iter(dataloader)
    global l_iteration, reducer
    
    log_epoch = {}
    ### Reused code from ends
    
    end = time.time()

    for i in range(0,300):
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
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

        # - forward
        output = model.forward(l_iteration)
           
        myoptimizers['FAN'].step()
                
        meters['losses']['rec'].update(model.loss['rec'].item(), bSize)
        l_iteration = l_iteration + 1

        ### Reused code from ends
        
        if i % 100 == 0:
            ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
            allimgs_deformed = None
            for (ii,imtmp) in enumerate(model.A['ImP'].to('cpu').detach()):
                hyper = imtmp.shape[0] >3
                if hyper:
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

            ### Reused code from ends
            
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

                plotter['losses']['rec'].log( l_iteration, model.loss['perp'].item() )
        
        ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019 
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
        ### Reused code from ends

    ### Reused code from begins https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019
    with open(args.folder + '/args_' + args.file[0:-8] + '_' + str(epoch) + '.pkl','wb') as f:
        pickle.dump(log_epoch,f)  

    avg_loss =  meters['losses']['rec'].avg
    ### Reused code from ends

    meters['losses']['rec'].reset()
    meters['losses']['perp'].reset()

    return avg_loss

if __name__ == '__main__':
    main()


