import torch
import rasterio

import numpy as np 
import torch.nn as nn 
from model_utils.checkpointing import load_ckpt_from_state_dict

from models.ssl import SSLHead
from models.SwinUNETR import SwinUNETR

model_2 = SwinUNETR(img_size=256 , in_channels=13 , out_channels=5 , )
# model = SSLHead()

model_2 = load_ckpt_from_state_dict("/home/amhamdi/Desktop/swinunetr_selfsupervised/checkpoint.pth",model_2 )


# with rasterio.open("/home/amhamdi/Desktop/s2_b/idx_0-256.tif") as data:
#     array = data.read()
#     array = array.astype(np.float32)
# array = torch.tensor(array)
# array = torch.unsqueeze(array,dim = 0)


# out = model_2(array)

# print(out.shape)