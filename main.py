import torch
import rasterio

import numpy as np 
import torch.nn as nn 
from model import SSL_Model

model = SSL_Model()

with rasterio.open("/home/amhamdi/Desktop/s2_b/idx_0-256.tif") as data:
    array = data.read()
    array = array.astype(np.float32)
array = torch.tensor(array)
array = torch.unsqueeze(array,dim = 0)


out = model(array)
# final_out = out[0]
# print("*"*30)
# print(out[2][-3].shape)
# exit()


print(out[0].shape)
print(out[1].shape)
print(out[2].shape)
