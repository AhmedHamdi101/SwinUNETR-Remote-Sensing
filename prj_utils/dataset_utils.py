import rasterio
import numpy as np
import torch
from torchvision.transforms import Normalize , ToTensor
from numpy import random
from torch.nn import Dropout

def rasterio_read_img(path , mean , std , max):
    
    # print(path)
    with rasterio.open(path) as s2_data:
        s2 = s2_data.read()
    s2 = s2.astype(np.float32)
    
    s2 = np.transpose(s2,(1,2,0))
    s2 = ToTensor()(s2)
    
    for i in range(len(s2)):
        s2[i] = s2[i] / float(max[i])

    s2 = normalize_tensor(mean , std , s2)
    return s2

def rasterio_read_img_meta(path ):
    
    # print(path)
    with rasterio.open(path) as s2_data:
        # s2 = s2_data.read()
        meta = s2_data.meta
   
    # s2 = normalize_tensor(mean , std , s2)

    return meta

def normalize_tensor(mean , std , tensor):
    for i in range(len(mean)):
        tensor[i] = (tensor[i] - mean[i]) / std[i]
    return tensor


def drop_channel(input):

    index = random.randint(len(input))
    sample_GT = torch.unsqueeze(input[index].detach().clone() , dim = 0)
    volume_GT = torch.cat([input[0:index] , input[index+1 :]  ])
    zeros = torch.zeros(input[index].shape)
    input[index]= zeros

    return input , sample_GT , volume_GT , index

def volume_inpainting(input):
    volume_inpainting_layer = Dropout(p=0.2)
    input = volume_inpainting_layer(input)
    return input
