import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
from prj_utils.main_utils import main_train , main_validate


from dataloader import SentinelDataloader
import numpy as np
import matplotlib.pyplot as plt 

torch.manual_seed(0)
# main_validate("0")
main_train("0")

# dataloaders = SentinelDataloader(root_img_dir="/home/amhamdi/Desktop/swinunetr_selfsupervised/cropped_data" , train_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/train_samples.csv",
#                                         valid_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv" , test_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv",
#                                         batch_size=6 , num_workers= 6)
# dataloaders.setup(stage="fit")

# train_dataloader = dataloaders.train_dataloader()
# validation_dataloader = dataloaders.val_dataloader()

# for batch in train_dataloader :


#     index = 0

#     input =  batch["X"][index].numpy()

#     positive_img = batch["X_P"][index].numpy()

#     negative_img = batch["X_N"][index].numpy()

#     image_GT = batch["Y"][index].numpy()

#     channel_GT = batch["Y_CHANNEL"][index].numpy()
#     volume_GT = batch["Y_INVOLUME"][index].numpy()

#     channel_idx = batch["CHANNEL_IDX"][index].numpy()
#     plt.figure()

#     print(channel_idx)
#     plt.subplot(1, 4, 1)
#     plt.imshow(input[0])
#     plt.subplot(1, 4, 2)
#     plt.imshow(positive_img[0])    
#     plt.subplot(1, 4, 3)
#     plt.imshow(negative_img[0])    
#     # plt.subplot(1, 4, 4)
#     # plt.imshow(volume_GT[-1])    
    
#     plt.show()
    
    
#     # print(volume_GT.shape)
    
#     # plt.show()
    
#     # print(image_GT.shape)
#     # plt.imshow(image_GT[0])
#     # plt.show()

#     exit()