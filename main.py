import comet_ml
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from models.ssl_ptl_module import SSL_PTLModule
from dataloader import SentinelDataloader
import pytorch_lightning as pl


torch.manual_seed(0)

logger = CometLogger(
            api_key="oZ01tNc0QwfzPcXODfwkCrDdL",
            workspace="amhamdi",
            # project_name=,  # Optional
            experiment_name="tmp",  # Optional
            experiment_key=None
        )


checkpoint_callback = ModelCheckpoint(
        dirpath="/home/amhamdi/Desktop/swinunetr_selfsupervised/checkpoints/",
        filename="checkpoint" + "-{epoch:02d}-{step:02d}-",
        save_on_train_epoch_end=True,
        save_top_k=-1,
        every_n_train_steps=500)

dataloaders = SentinelDataloader(root_img_dir="/home/amhamdi/Desktop/swinunetr_selfsupervised/cropped_data" , train_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/train_samples.csv",
                                    valid_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv" , test_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv",
                                    batch_size=12 , num_workers= 12)
dataloaders.setup(stage="fit")

train_dataloader = dataloaders.train_dataloader()
validation_dataloader = dataloaders.val_dataloader()
model = SSL_PTLModule(0.01)
trainer = pl.Trainer(   accelerator="gpu",
                        devices=1,
                        num_nodes=1,
                        max_epochs=100,
                        logger=logger,
                        callbacks=checkpoint_callback,
                        # val_check_interval=0.001,
                        log_every_n_steps=5)
trainer.fit(model=model,train_dataloaders = train_dataloader , val_dataloaders=validation_dataloader)      
# dda = SentinelDataset("/home/amhamdi/Desktop/swinunetr_selfsupervised/cropped_data/" , "/home/amhamdi/Desktop/swinunetr_selfsupervised/samples.csv")

# for i in dda:
#     print(torch.unique(i["X"]))
#     exit()

# import torch
# import rasterio

# import numpy as np 
# import torch.nn as nn 
# from model_utils.checkpointing import load_ckpt_from_state_dict

# from models.ssl import SSLHead
# from models.SwinUNETR import SwinUNETR

# model_2 = SwinUNETR(img_size=256 , in_channels=13 , out_channels=5 , )
# # model = SSLHead()

# model_2 = load_ckpt_from_state_dict("/home/amhamdi/Desktop/swinunetr_selfsupervised/checkpoint.pth",model_2 )


# with rasterio.open("/home/amhamdi/Desktop/s2_b/idx_0-256.tif") as data:
#     array = data.read()
#     array = array.astype(np.float32)
# array = torch.tensor(array)
# array = torch.unsqueeze(array,dim = 0)


# out = model_2(array)

# print(out.shape)
