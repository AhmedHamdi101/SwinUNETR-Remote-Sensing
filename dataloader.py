
import torch

from dataset import SentinelDataset
from torch.utils.data import DataLoader 
from pytorch_lightning import LightningDataModule


class SentinelDataloader(LightningDataModule):

    def __init__(self, root_img_dir , train_df, valid_df, test_df,
     batch_size = 32 , Transform = None , num_workers = 1,
     normalize_values = [(0.14298654, 0.074617684, 0.08758479 , 0.08413466 , 0.12160285 , 0.15589541 , 0.1592145 , 0.13745776,0.1438589, 0.09315324 , 0.0069550727 , 0.12110794 , 0.0861285),
                     (0.033589855,0.026032435, 0.041410778 ,  0.059052367 , 0.07244488 , 0.06593627 , 0.06685208 , 0.059768923 ,  0.06090644 ,  0.051596936 , 0.01042658 , 0.076781936 , 0.06678088)],
                    max = [11158.0 ,19059.0,16714.0,18522.0, 14644.0 , 15888.0 , 17688.0 , 19630.0 , 20467.0 , 10447.0 , 2075.0 , 19337.0 , 20364.0]):
        super().__init__()

        self.img_path = root_img_dir

        self.train_df = train_df

        self.valid_df = valid_df
        
        self.test_df = test_df

        self.batch_size = batch_size
        self.transform = Transform

        self.num_workers = num_workers
        
        self.normalize_values = normalize_values
        self.images_max = max
   
    def setup(self,stage=None):
        if stage == "fit" or stage == None:
            self.train_dataset = SentinelDataset(self.img_path,self.train_df,
                                                            transform=self.transform,
                                                            normalize_values= self.normalize_values,
                                                            max = self.images_max)
            self.valid_dataset = SentinelDataset(self.img_path,self.valid_df,
                                                            transform=None ,
                                                            normalize_values= self.normalize_values,
                                                            max = self.images_max)


        if stage == "test":
            self.test_dataset = SentinelDataset(self.img_path,self.valid_df,
                                                            transform=None,
                                                            normalize_values= self.normalize_values,
                                                            max = self.images_max)

    def train_dataloader(self):
        return DataLoader(self.train_dataset , batch_size=self.batch_size,
                        shuffle = True,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset , batch_size=self.batch_size,
                        shuffle = False,num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset , batch_size=self.batch_size,
                        shuffle = False,num_workers=self.num_workers)


