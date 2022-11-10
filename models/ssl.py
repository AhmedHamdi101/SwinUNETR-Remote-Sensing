import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self, upsample="vae", dim=768):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, 2)
        window_size = ensure_tuple_rep(7, 2)

        
        self.swinViT = SwinViT(
            in_chans=13,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=2,
        )

        self.inpainting_head = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(384, 192, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),


            nn.ConvTranspose2d(192, 96, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(96, 48, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.ConvTranspose2d(48, 24, 5 ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(24, 12, 5 ),
            nn.LeakyReLU()
            )

        self.channel_reconstruction = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(384, 192, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),


            nn.ConvTranspose2d(192, 96, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(96, 48, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.ConvTranspose2d(48, 24, 5 ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(24, 12, 3 ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 1, 3 ),
            nn.LeakyReLU()
            )

        self.triplet_head = nn.Sequential(
            nn.Conv2d(768,384,3),
            nn.LeakyReLU(),
            nn.Conv2d(384,192,3),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(3072,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256)
        )


    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        inpainting_output = self.inpainting_head(x_out)
        reconstruction_output = self.channel_reconstruction(x_out)
        triplet_output = self.triplet_head(x_out)
        return [inpainting_output , reconstruction_output , triplet_output]
        
