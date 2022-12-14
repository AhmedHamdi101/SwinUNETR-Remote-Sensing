import os
import torch
import rasterio
import numpy as np
import math
from torchvision.transforms import Normalize , Compose
from prj_utils.dataset_utils import rasterio_read_img_meta

def load_ckpt_from_state_dict(path,model):
    try:
        state_dict = torch.load(path)
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        if "swin_vit" in list(state_dict.keys())[0]:
            print("Tag 'swin_vit' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=False)
        print("Using pretrained self-supervised Swin UNETR backbone weights !")
        return model
    except ValueError:
        raise ValueError("Self-supervised pre-trained weights not available ")

def save_outputs(paths, channel_GT , channel_pred , inpainting_GT , inpainting_pred , idx,save_path ):
    max = [11158.0 ,19059.0,16714.0,18522.0, 14644.0 , 15888.0 , 17688.0 , 19630.0 , 20467.0 , 10447.0 , 2075.0 , 19337.0 , 20364.0]
    
    mean_0 = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
    std_0 = [ 1/0.033589855,1/0.026032435, 1/0.041410778 ,  1/0.059052367 , 1/0.07244488 , 1/0.06593627 , 1/0.06685208 , 1/0.059768923 ,  1/0.06090644 ,  1/0.051596936 , 1/0.01042658 , 1/0.076781936 , 1/ 0.06678088 ]
    
    mean_1 = [ -0.14298654, -0.074617684, -0.08758479 , -0.08413466 , -0.12160285 , -0.15589541 , -0.1592145 , -0.13745776,-0.1438589, -0.09315324 , -0.0069550727 , -0.12110794 , -0.0861285]
    std_1 = [ 1., 1. , 1., 1., 1. , 1., 1., 1. , 1., 1., 1.,1., 1. ]




    for i in range(len(paths)):

        path = paths[i]
        channel_idx = idx[i]

        folder_1 = os.path.basename(os.path.dirname(paths[i]))

        invTrans_channel = Compose([ Normalize(mean = mean_0[channel_idx],
                                            std  =  std_0[channel_idx]),
                                    Normalize( mean = mean_1[channel_idx],
                                            std =   std_1[channel_idx]),
                                ])
        invTrans_inpainting = Compose([ Normalize(  mean = mean_0[: channel_idx] + mean_0[channel_idx+1 :],
                                                    std = std_0[: channel_idx] + std_0[channel_idx+1 :]),
                                        Normalize(  mean = mean_1[: channel_idx] + mean_1[channel_idx+1 :],
                                                    std = std_1[: channel_idx] + std_1[channel_idx+1 :])
                                ])    



        # channel_GT_sample = invTrans_channel(channel_GT[i])
        channel_GT_sample = invTrans_channel(channel_GT[i]) * max[channel_idx]
        channel_GT_sample = channel_GT_sample.cpu().detach().numpy()
        

        inpainting_GT_sample = invTrans_inpainting(inpainting_GT[i])
        inpainting_GT_sample = multiply_channelwise_by_max(inpainting_GT_sample , max[:channel_idx]+max[channel_idx+1:])
        inpainting_GT_sample = inpainting_GT_sample.cpu().detach().numpy()

        # channel_pred_sample = invTrans_channel(channel_pred[i]) 
        channel_pred_sample = invTrans_channel(channel_pred[i]) * max[channel_idx]
        channel_pred_sample = channel_pred_sample.cpu().detach().numpy()
        
        inpainting_pred_sample = invTrans_inpainting(inpainting_pred[i])
        inpainting_pred_sample = multiply_channelwise_by_max(inpainting_pred_sample , max[:channel_idx]+max[channel_idx+1:])
        inpainting_pred_sample = inpainting_pred_sample.cpu().detach().numpy()

        name = folder_1+"_"+os.path.basename(path)
        meta = rasterio_read_img_meta(path=path)

        # print(meta)
        meta['count'] = 1
        # print(channel_pred.shape)

        with rasterio.open(os.path.join(save_path,name+"_"+str(channel_idx.item())+"_CHANNEL_PRED.tif"), 'w', **meta) as dst:
            dst.write(channel_pred_sample)
        meta['count'] = 12
        with rasterio.open(os.path.join(save_path,name+"_INPAINTING_PRED.tif"), 'w', **meta) as dst:
            dst.write(inpainting_pred_sample)
        meta['count'] = 1
        with rasterio.open(os.path.join(save_path,name+"_"+str(channel_idx.item())+"_CHANNEL_GT.tif"), 'w', **meta) as dst:
            dst.write(channel_GT_sample)
        meta['count'] = 12
        with rasterio.open(os.path.join(save_path,name+"_INPAINTING_GT.tif"), 'w', **meta) as dst:
            dst.write(inpainting_GT_sample)


def multiply_channelwise_by_max(tensor , max ):
    
    for i in range(len(max)):
        tensor[i] = tensor[i] *max[i]
    return tensor 


def load_from_pretrained_weights(model , path = None , gpu = None):
    if path != None:
        if  gpu == None:
            weights = torch.load(path,map_location=torch.device('cpu'))
        else:
            weights = torch.load(path)
        model.load_state_dict (weights["state_dict"])
    return model

def calculate_psnr(img1, img2, border=0):

    if not img1.shape == img2.shape:
        print(img1.shape,img2.shape)
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnrb(img1, img2, border=0):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        img1, img2 = np.expand_dims(img1, 2), np.expand_dims(img2, 2)

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

def unormalize( channel_GT , channel_pred , inpainting_GT , inpainting_pred  ,idx):
    device = channel_pred.device
   
    channel_GT_temp = torch.tensor([])
    channel_pred_temp = torch.tensor([])
    
    inpainting_GT_temp = torch.tensor([])
    inpainting_pred_temp = torch.tensor([])
    


    max_values = [11158.0 ,19059.0,16714.0,18522.0, 14644.0 , 15888.0 , 17688.0 , 19630.0 , 20467.0 , 10447.0 , 2075.0 , 19337.0 , 20364.0]
    
    mean_0 = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
    std_0 = [ 1/0.033589855,1/0.026032435, 1/0.041410778 ,  1/0.059052367 , 1/0.07244488 , 1/0.06593627 , 1/0.06685208 , 1/0.059768923 ,  1/0.06090644 ,  1/0.051596936 , 1/0.01042658 , 1/0.076781936 , 1/ 0.06678088 ]
    
    mean_1 = [ -0.14298654, -0.074617684, -0.08758479 , -0.08413466 , -0.12160285 , -0.15589541 , -0.1592145 , -0.13745776,-0.1438589, -0.09315324 , -0.0069550727 , -0.12110794 , -0.0861285]
    std_1 = [ 1., 1. , 1., 1., 1. , 1., 1., 1. , 1., 1., 1.,1., 1. ]


    for i in range(len(idx)):

        channel_idx = idx[i]
        invTrans_channel = Compose([ Normalize(mean = mean_0[channel_idx],
                                            std  =  std_0[channel_idx]),
                                    Normalize( mean = mean_1[channel_idx],
                                            std =   std_1[channel_idx]),
                                ])
        invTrans_inpainting = Compose([ Normalize(  mean = mean_0[: channel_idx] + mean_0[channel_idx+1 :],
                                                    std = std_0[: channel_idx] + std_0[channel_idx+1 :]),
                                        Normalize(  mean = mean_1[: channel_idx] + mean_1[channel_idx+1 :],
                                                    std = std_1[: channel_idx] + std_1[channel_idx+1 :])
                                ])    

        channel_GT_sample = invTrans_channel(channel_GT[i]) * max_values[channel_idx]
        channel_pred_sample = invTrans_channel(channel_pred[i]) * max_values[channel_idx]       

        inpainting_GT_sample = invTrans_inpainting(inpainting_GT[i])
        inpainting_GT_sample = multiply_channelwise_by_max(inpainting_GT_sample , max_values[:channel_idx]+max_values[channel_idx+1:])

        
        inpainting_pred_sample = invTrans_inpainting(inpainting_pred[i])
        inpainting_pred_sample = multiply_channelwise_by_max(inpainting_pred_sample , max_values[:channel_idx]+max_values[channel_idx+1:])
    
        if len(channel_GT_temp) == 0:

            channel_GT_temp = torch.unsqueeze(channel_GT_sample , dim = 0)
            channel_pred_temp = torch.unsqueeze(channel_pred_sample, dim = 0)

            inpainting_GT_temp = torch.unsqueeze(inpainting_GT_sample, dim = 0)
            inpainting_pred_temp = torch.unsqueeze(inpainting_pred_sample, dim = 0)
        else:

            channel_GT_temp = torch.cat([channel_GT_temp , torch.unsqueeze(channel_GT_sample, dim = 0)] , dim = 0)
            channel_pred_temp = torch.cat([channel_pred_temp , torch.unsqueeze(channel_pred_sample, dim = 0)] , dim = 0)


            inpainting_GT_temp = torch.cat([inpainting_GT_temp , torch.unsqueeze(inpainting_GT_sample, dim = 0)] , dim = 0)
            inpainting_pred_temp = torch.cat([inpainting_pred_temp , torch.unsqueeze(inpainting_pred_sample, dim = 0)] , dim = 0)

    channel_GT_temp = channel_GT_temp.to(torch.device(device))
    channel_pred_temp = channel_pred_temp.to(torch.device(device)) 

    inpainting_GT_temp = inpainting_GT_temp.to(torch.device(device)) 
    inpainting_pred_temp = inpainting_pred_temp.to(torch.device(device)) 


    return channel_GT_temp , channel_pred_temp , inpainting_GT_temp ,  inpainting_pred_temp