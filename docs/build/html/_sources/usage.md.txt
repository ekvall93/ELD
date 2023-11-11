# Usage

## Installation

To install ELD:

```console
(.venv) $ pip install .
```

Then you have to install ```torch==1.10.0```, ```torchvision==1.10.0```, and ```torchaudio==1.10.0``` with the right CUDA version from [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/). Here's an example:

```console
(.venv) $ pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training

For training [Visdom](https://github.com/fossasia/visdom) is used to easily following the training process. To start a Visdom server, do the following

```console
(.venv) $ python -m visdom.server -port 9006
```

You can train models using different types of data: unimodal, 3D, and multimodal. The following sections describe how to use each type.

### Unimodal

To train a model using unimodal data, use the ```--model unimodal``` flag. Here is an example command:

```
eld-train --elastic_sigma 5 --cuda 1 --port 9006 --data_path ../marcoAnalysis/MOB_HE/ --npts 16 --o scratch --elastic True --step_size 5 --ws 0 --gamma 0.9 --angle 8 --model unimodal
```

### 3D Data
To train a model on 3D tissue landmarks, use the ```--model 3d``` flag. The image files for this model should be named ```0.png, 1.png,...,n.png``` where the number indicates the image's position in the stack. Here is an example command:


```
eld-train --elastic_sigma 5 --cuda 1 --port 9006 --data_path ../marcoAnalysis/CODA_prostate/ --npts 16 --o scratch --elastic True --step_size 5 --ws 0 --gamma 0.9 --angle 8 --model 3d
````


### Multimodal Data
To train a model using multimodal data, use the ```--model multimodal``` flag. The corresponding files should end with ```*_mod1.png``` or ```*_mod2.png```, which indicate their respective modality. Here is an example command:


```
eld-train --elastic_sigma 5 --cuda 1 --port 9006 --data_path ../marcoAnalysis/CODA_prostate/ --npts 16 --o scratch --elastic True --step_size 5 --ws 0 --gamma 0.9 --angle 8 --model multimodal
````
