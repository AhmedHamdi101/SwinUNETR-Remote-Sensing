import torch
import torch.nn as nn 
# from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from transformers import SwinConfig, SwinModel




class SSL_Model(nn.Module):
    def __init__(self):
        super(SSL_Model,self ).__init__()
        configuration = SwinConfig(image_size=256,num_channels=13,depths=[2,2,2,2],num_heads=[3,6,12,24])
        self.backbone = SwinModel(configuration)

        self.unflatten_layer = nn.Unflatten(dim=-1 , unflattened_size=(28,28))

        self.inpainting_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(32, 16, 3 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),


            nn.ConvTranspose2d(16, 12, 3 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(12, 12, 3 ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 12, 3 ),
            nn.LeakyReLU()
            )

        self.channel_reconstruction = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(32, 16, 3 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),


            nn.ConvTranspose2d(16, 12, 3 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(12, 12, 3 ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 1, 3 ),
            nn.LeakyReLU()
            )

        self.triplet_head = nn.Sequential(
            nn.Conv2d(64,64,7,stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,5,stride=2),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(2048,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
        )


    def forward(self,x):
        backbone_output = self.backbone(x,output_hidden_states=True,return_dict=True)
        reshaped_output = self.reshape_output(backbone_output[0])
        inpainting_output = self.inpainting_head(reshaped_output)
        reconstruction_output = self.channel_reconstruction(reshaped_output)
        triplet_output = self.triplet_head(reshaped_output)

        return [inpainting_output , reconstruction_output , triplet_output]

    def reshape_output(self,tensor):
        zeros = torch.zeros((tensor.shape[0] , tensor.shape[1] , 16))
        tensor = torch.cat([tensor,zeros],dim=-1)
        tensor = self.unflatten_layer(tensor)
        return tensor

