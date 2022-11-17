import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self, dim=768):
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
            nn.ConvTranspose2d(dim, dim//2, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(dim//2, dim//4, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),


            nn.ConvTranspose2d(dim//4, dim//8, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(dim//8, dim//16, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.ConvTranspose2d(dim//16, dim//32, 5 ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(dim//32, dim//64, 5 ),
            nn.LeakyReLU()
            )

        self.channel_reconstruction = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(dim//2, dim//4, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),


            nn.ConvTranspose2d(dim//4, dim//8, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.ConvTranspose2d(dim//8, dim//16, 5 ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.ConvTranspose2d(dim//16, dim//32, 5 ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(dim//32, dim//64, 3 ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(dim//64, 1, 3 ),
            nn.LeakyReLU()
            )

        self.triplet_head = nn.Sequential(
            nn.Conv2d(dim,dim//2,3),
            nn.LeakyReLU(),
            nn.Conv2d(dim//2,dim//4,3),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(dim*4,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256)
        )


        self.swinViT.apply(self.init_weights)
        self.inpainting_head.apply(self.init_weights)
        self.channel_reconstruction.apply(self.init_weights)
        self.triplet_head.apply(self.init_weights)


    def init_weights(self,m):
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

        if isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight)


            



    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        inpainting_output = self.inpainting_head(x_out)
        reconstruction_output = self.channel_reconstruction(x_out)
        triplet_output = self.triplet_head(x_out)
        return [inpainting_output , reconstruction_output , triplet_output]
        
