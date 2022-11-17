import os
import ast
import torch
import numpy as np
import pandas as pd
from numpy import random

from torch.utils.data import Dataset


from prj_utils.dataset_utils import rasterio_read_img , normalize_tensor , drop_channel , volume_inpainting

class SentinelDataset(Dataset):
    
    def __init__(self, root_img_dir, samples_df_path ,
                    transform = None, normalize_values= [[0.14298654, 0.074617684, 0.08758479 , 0.08413466 , 0.12160285 , 0.15589541 , 0.1592145 , 0.13745776,0.1438589, 0.09315324 , 0.0069550727 , 0.12110794 , 0.0861285],
                     [0.033589855,0.026032435, 0.041410778 ,  0.059052367 , 0.07244488 , 0.06593627 , 0.06685208 , 0.059768923 ,  0.06090644 ,  0.051596936 , 0.01042658 , 0.076781936 , 0.06678088]],
                    max = [11158.0 ,19059.0,16714.0,18522.0, 14644.0 , 15888.0 , 17688.0 , 19630.0 , 20467.0 , 10447.0 , 2075.0 , 19337.0 , 20364.0]):
 
        self.samples_df = pd.read_csv(samples_df_path)
        self.root_img_dir = root_img_dir
        self.transform = transform
        self.normalize_values = normalize_values
        self.max = max

        assert os.path.exists(self.root_img_dir) , self.root_img_dir
        
        self.samples = self.read_samples(self.samples_df)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
       
        sample = self.samples[index]
        path = sample["X"]
        main_img = rasterio_read_img(path , self.normalize_values[0], self.normalize_values[1] , self.max)
        main_img_channel , channel_GT , volume_GT , channel_index = drop_channel(main_img)
        main_img_volume_channel = volume_inpainting(main_img_channel)        
        path = sample["positive_im"]
        positive_img = rasterio_read_img(path , self.normalize_values[0], self.normalize_values[1], self.max)
        
        path = sample["negative_im"]
        negative_img = rasterio_read_img(path , self.normalize_values[0], self.normalize_values[1], self.max)
        
        
        sample = {  "X": main_img_volume_channel,

                    "X_P":positive_img, "X_P_PATH": sample["positive_im"] ,
                    "X_N": negative_img, "X_N_PATH": sample["negative_im"],

                    "Y":main_img , "Y_PATH":sample["X"] , "Y_CHANNEL": channel_GT , "Y_INVOLUME":volume_GT , "CHANNEL_IDX":channel_index
                    }

        return sample


    def read_samples(self,samples_df):
        samples = []
        for i in range(len(samples_df)):
            img = os.path.join(self.root_img_dir,samples_df["sample"][i])
            positive_id =  os.path.basename(samples_df["sample"][i])

            overlapped_dict = ast.literal_eval(samples_df["overlapped_samples"][i])
        
            
            img_ov_1 = os.path.join(self.root_img_dir,overlapped_dict["1"])
            img_ov_2 = os.path.join(self.root_img_dir,overlapped_dict["2"])
            img_ov_3 = os.path.join(self.root_img_dir,overlapped_dict["3"])
            img_over = [] 

            img_over.append(img_ov_1)
            img_over.append(img_ov_2)
            img_over.append(img_ov_3)
            
            index = random.randint(len(samples_df))
            negative_id = os.path.basename(samples_df["sample"][index])
            while index == i or negative_id == positive_id:
                index = random.randint(len(samples_df))
                negative_id = os.path.basename(samples_df["sample"][index])
            negative_img = os.path.join(self.root_img_dir,samples_df["sample"][index])
            

            index = random.randint(len(img_over))
            

            samples.append({"X": img, "positive_im":img_over[index] ,"negative_im" : negative_img })

        return samples 

