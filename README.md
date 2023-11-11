<h1 align="center">ELD – Effortless Landmark Detection</h1>


<p align="center">
<img src="./ELD.png" alt="ELD logo"/>
</p>


This repository contains notebooks and code for the ELD project. The main goal is to provide a toolset for training various models on different types of data.


## Dataset
The data used in these notebooks is publicly available and can be accessed from the following link: ELD project on Figshare.


## How to Train Models
You can train models using different types of data: unimodal, 3D, and multimodal. The following sections describe how to use each type.


### Unimodal Data
To train a model using unimodal data, use the --model unimodal flag. Here is an example command:




```
python train.py --elastic_sigma 5 --cuda 1 --port 9006 --data_path ../marcoAnalysis/MOB_HE/ --npts 16 --o scratch --elastic True --step_size 5 --ws 0 --gamma 0.9 --angle 8 --model unimodal
```


### 3D Data
To train a model on 3D tissue landmarks, use the --model 3d flag. The image files for this model should be named 0.png, 1.png,...,n.png where the number indicates the image's position in the stack. Here is an example command:


```
python train.py --elastic_sigma 5 --cuda 1 --port 9006 --data_path ../marcoAnalysis/CODA_prostate/ --npts 16 --o scratch --elastic True --step_size 5 --ws 0 --gamma 0.9 --angle 8 --model 3d
````


### Multimodal Data
To train a model using multimodal data, use the --model multimodal flag. The corresponding files should end with *_mod1.png or *_mod2.png, which indicate their respective modality. Here is an example command:


```
python train.py --elastic_sigma 5 --cuda 1 --port 9006 --data_path ../marcoAnalysis/CODA_prostate/ --npts 16 --o scratch --elastic True --step_size 5 --ws 0 --gamma 0.9 --angle 8 --model multimodal
````


Please adjust the parameters according to your needs.




## Credits


This project makes use of the methods described in the paper [Object Landmark Discovery Through Unsupervised Adaptation](http://papers.nips.cc/paper/9505-object-landmark-discovery-through-unsupervised-adaptation.pdf) by Enrique Sanchez and Georgios Tzimiropoulos, as presented at the Neural Information Processing Systems (NeurIPS) conference in 2019.


A portion of our implementation is based on the code found in the following GitHub repository: [SAIC-Unsupervised-landmark-detection-NeurIPS2019](https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019).


We would like to express our gratitude to the authors for their valuable work and for making their code available for the community.
