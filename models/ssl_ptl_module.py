import os
import torch 
import pytorch_lightning as pl
from torch.nn import TripletMarginLoss , L1Loss 
from torchmetrics import PeakSignalNoiseRatio , StructuralSimilarityIndexMeasure

from models.ssl import SSLHead 
from prj_utils.model_utils import save_outputs
class SSL_PTLModule(pl.LightningModule):

    def __init__(self,learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

        self.model = SSLHead()

        self.triplet_loss = TripletMarginLoss(margin=1.0 , p=2) 
        self.l1_loss = L1Loss()
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
    
    
    def forward(self,batch):
        prediction = self.model(batch)
        return prediction


    def calculate_loss(self,volume_GT, channel_GT , model_predictions , triplet_positive , triplet_negative):

        inpainting_output , reconstruction_output , triplet_output = model_predictions[0] , model_predictions[1] , model_predictions[2]

        triplet_loss_value = self.triplet_loss(triplet_output , triplet_positive , triplet_negative)
        
        inpainting_loss_value = self.l1_loss(inpainting_output , volume_GT)
        
        reconstruction_loss_value = self.l1_loss(reconstruction_output , channel_GT)

        # loss = (triplet_loss_value + inpainting_loss_value + reconstruction_loss_value) / 3
        return triplet_loss_value , inpainting_loss_value , reconstruction_loss_value
            
    def training_step(self, batch , step):
         
        batch_size = len(batch)
        self.input =  batch["X"]

        self.positive_img = batch["X_P"]
        self.positive_path = batch["X_P_PATH"]
        
        self.negative_img = batch["X_N"]
        self.negative_path = batch["X_N_PATH"]

        self.image_GT = batch["Y"]
        self.image_GT_path = batch["Y_PATH"]
        self.channel_GT = batch["Y_CHANNEL"]
        self.volume_GT = batch["Y_INVOLUME"]

        predictions = self(self.input)
        
        inpainting_output , reconstruction_output , _ = predictions[0] , predictions[1] , predictions[2]
       
        triplet_positive = self(self.positive_img)[2]
        triplet_negative = self(self.negative_img)[2]

        triplet_loss_value , inpainting_loss_value , reconstruction_loss_value = self.calculate_loss( self.volume_GT , self.channel_GT , predictions , triplet_positive , triplet_negative)
        loss = (triplet_loss_value + inpainting_loss_value + reconstruction_loss_value) / 3

        
        psnr_channel = self.psnr(reconstruction_output , self.channel_GT)
        psnr_inpainting = self.psnr(inpainting_output , self.volume_GT)
        

        # ssim_channel = self.ssim(reconstruction_output , self.channel_GT)
        # ssim_inpainting = self.ssim(inpainting_output , self.volume_GT)

        self.log("Train_Total_loss", loss ,  batch_size=batch_size)
        self.log("Train_Triplet_loss", triplet_loss_value ,  batch_size=batch_size)
        self.log("Train_Inpainting_loss", inpainting_loss_value ,  batch_size=batch_size)
        self.log("Train_Channel_loss", reconstruction_loss_value ,  batch_size=batch_size)
        
        self.log("Train_Channel_PSNR",psnr_channel ,  batch_size=batch_size)
        self.log("Train_Inpainting_PSNR",psnr_inpainting,  batch_size=batch_size)

        # self.log("Train_Channel_SSIM",ssim_channel ,  batch_size=batch_size)
        # self.log("Train_Inpainting_SSIM",ssim_inpainting,  batch_size=batch_size)
        
        return loss

    def validation_step(self, batch , batch_idx):
        batch_size = len(batch)

        self.input =  batch["X"]

        self.positive_img = batch["X_P"]
        self.positive_path = batch["X_P_PATH"]
        
        self.negative_img = batch["X_N"]
        self.negative_path = batch["X_N_PATH"]

        self.image_GT = batch["Y"]
        self.image_GT_path = batch["Y_PATH"]
        self.channel_GT = batch["Y_CHANNEL"]
        self.volume_GT = batch["Y_INVOLUME"]

        self.channel_idx = batch["CHANNEL_IDX"]
        
        predictions = self(self.input)
        
        inpainting_output , reconstruction_output , _ = predictions[0] , predictions[1] , predictions[2]
       
        triplet_positive = self(self.positive_img)[2]
        triplet_negative = self(self.negative_img)[2]

        triplet_loss_value , inpainting_loss_value , reconstruction_loss_value = self.calculate_loss( self.volume_GT , self.channel_GT , predictions , triplet_positive , triplet_negative)
        loss = (triplet_loss_value + inpainting_loss_value + reconstruction_loss_value) / 3
        
        psnr_channel = self.psnr(reconstruction_output , self.channel_GT)
        psnr_inpainting = self.psnr(inpainting_output , self.volume_GT)
        
        # ssim_channel = self.ssim(reconstruction_output , self.channel_GT)
        # ssim_inpainting = self.ssim(inpainting_output , self.volume_GT)

        self.log("Validation_Total_loss", loss , batch_size=batch_size)
        self.log("Validation_Triplet_loss", triplet_loss_value ,  batch_size=batch_size)
        self.log("Validation_Inpainting_loss", inpainting_loss_value ,  batch_size=batch_size)
        self.log("Validation_Channel_loss", reconstruction_loss_value ,  batch_size=batch_size)

        self.log("Validation_Channel_PSNR",psnr_channel , batch_size=batch_size)
        self.log("Validation_Inpainting_PSNR",psnr_inpainting, batch_size=batch_size)
        
        # self.log("Validation_Channel_SSIM",ssim_channel ,  batch_size=batch_size)
        # self.log("Validation_Inpainting_SSIM",ssim_inpainting,  batch_size=batch_size)
        
        # save_outputs(self.image_GT_path , self.channel_GT , reconstruction_output , self.volume_GT , inpainting_output , self.channel_idx,"/home/amhamdi/Desktop/swinunetr_selfsupervised/Validation_output")


        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer
