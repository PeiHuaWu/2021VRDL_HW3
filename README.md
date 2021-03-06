# 2021VRDL_HW3-Instance Segmentation on Nuclei Datasets

This project is part of a series of projects for the course Selected Topics in Visual Recognition using Deep Learning. This Repository gathers the code for instance segmentation model.

## Environment
```
Google Colab
```

## Requirements

To install the following modules:
```
!pip install mmcv
!pip install pyyaml==5.1
```

Furthermore, please install detectron2 that matches the following pytorch version:
```
torch:  1.10 ; cuda:  cu111
```

```
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
```

## Data Preprocessing

Run this command to generate train.json in coco format from mask images.
```
!python imgs_to_coco.py
```

## Training

First, run the following command and download [train_all.json](https://drive.google.com/file/d/1s4SxQaStL5QRGww33ainXDgX9gbAcDTU/view?usp=sharing), and then put it in the folder "2021VRDL_HW3" :
```
!git clone https://github.com/PeiHuaWu/2021VRDL_HW3.git
%cd /content/2021VRDL_HW3
```
To train the model in the paper, run this command 
```
!python train.py   
```

## Testing

To evaluate my model on detectron2, please refer to [inference.ipynb](https://github.com/PeiHuaWu/2021VRDL_HW3/blob/main/inference.ipynb). You can download the required files in this python code, it's needless to download and upload the files by yourself.


## Files for downloading

You can also download the file here:

- [The file of model_final.pth](https://drive.google.com/file/d/1_BmSqW-x6eav4lXSDByWLXmka7X5AEIT/view?usp=sharing)
