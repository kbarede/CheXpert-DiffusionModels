import torch
from torch import nn
import math
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math


''' ---------------------------------------------------------------------------------------------------------------- '''
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.act = nn.SiLU()

        if in_ch != out_ch: self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else: self.residual_conv = nn.Identity()

    def forward(self, x, t):
        # First convolution
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add timestep information
        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]
        h = h + time_emb

        # Second convolution
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)

''' ---------------------------------------------------------------------------------------------------------------- '''
class AttentionBlock(nn.Module):
    """
    Self-attention over spatial positions
    This layer is supposed to help the network in making some spatial order like left vs right or up and down.
    So it learns spatial relationships between features.
    for example, it could help learning the the heart is always in the mid-left, and the ribs are somewhat symmetric 
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True # Batch comes first rather then seq
        )

    def forward(self, x):
        # Featuer map x:[B,C,H,W] -> [B,H*W,C] - atteionen layer need sequence rather than H and W
        B, C, H, W = x.shape
        h = self.norm(x)
        h = (h.flatten(2).transpose(1, 2))
        # cehck what postion is related to which postion - h=attention_output, _=attention_weights:
        h, _ = self.attention(h, h, h, need_weights=False) 
        # [B,H*W,C] -> [B,C,H,W]
        h = (h.transpose(1, 2).reshape(B, C, H, W))
        # residual attention
        return x + h
''' ---------------------------------------------------------------------------------------------------------------- '''
class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x
''' ---------------------------------------------------------------------------------------------------------------- '''
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

''' ---------------------------------------------------------------------------------------------------------------- '''
class ImprovedUnet(nn.Module):

    def __init__(self):
        super().__init__()
        image_channels = 1
        time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4,time_emb_dim)
        )
        # INPUT 128 × 128
        self.conv0 = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        
        # Level 1 - 128 × 128 channels = 64
        self.res1_1 = ResBlock(64, 64, time_emb_dim)
        self.res1_2 = ResBlock(64, 64, time_emb_dim)
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        
        # Level 2 - 64 × 64, channels = 128
        self.res2_1 = ResBlock(128, 128, time_emb_dim)
        self.res2_2 = ResBlock(128, 128, time_emb_dim)
        self.down2 = nn.Conv2d( 128, 256, kernel_size=4, stride=2, padding=1)
        
        # Level 3 - 32 × 32, channels = 256
        self.res3_1 = ResBlock(256, 256, time_emb_dim)
        self.res3_2 = ResBlock(256, 256, time_emb_dim)
        self.down3 = nn.Conv2d(256, 512, kernel_size=4,stride=2, padding=1)
        
        # Level 4 - 16 × 16 channels = 512
        self.res4_1 = ResBlock(512, 512, time_emb_dim)
        self.res4_2 = ResBlock(512, 512, time_emb_dim)
        # ATTENTION AT 16 × 16
        self.attn4 = AttentionBlock(512, num_heads=8)

        # down Level 16 × 16 -> 8 × 8
        self.down4 = nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1)
        # BOTTLENECK 8 × 8 - channels = 768
        self.bot1 = ResBlock(768, 768, time_emb_dim)
        self.bot_attention = AttentionBlock(768, num_heads=8)
        self.bot2 = ResBlock(768, 768, time_emb_dim)
        
        # UP Level 4 -  8 × 8 -> 16 × 16
        self.up4 = Upsample(768, 512)
        # 512 decoder + 512 skip = 1024
        self.up_res4_1 = ResBlock(1024, 512, time_emb_dim)
        self.up_res4_2 = ResBlock(512, 512, time_emb_dim)
        self.up_attn4 = AttentionBlock(512, num_heads=8)
        
        # UP Level 3, 16 × 16 -> 32 × 32
        self.up3 = Upsample(512, 256)
        self.up_res3_1 = ResBlock(512, 256, time_emb_dim)
        self.up_res3_2 = ResBlock(256, 256, time_emb_dim)
        
        # UP Level 2, 32 × 32 -> 64 × 64
        self.up2 = Upsample(256, 128)
        self.up_res2_1 = ResBlock(256, 128, time_emb_dim)
        self.up_res2_2 = ResBlock(128, 128, time_emb_dim)
        
        # up level 1 - 64 × 64 -> 128 × 128
        self.up1 = Upsample(128, 64)
        self.up_res1_1 = ResBlock(128, 64, time_emb_dim)
        self.up_res1_2 = ResBlock(64, 64, time_emb_dim)
        
        # Output
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_act = nn.SiLU()
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        # Level 1 - 128 × 128
        x = self.conv0(x)
        x = self.res1_1(x, t)
        x = self.res1_2(x, t)
        skip1 = x

        # Level 2 - 64 × 64
        x = self.down1(x)
        x = self.res2_1(x, t)
        x = self.res2_2(x, t)
        skip2 = x

        # Level 3 - 32 × 32
        x = self.down2(x)
        x = self.res3_1(x, t)
        x = self.res3_2(x, t)
        skip3 = x

        # Level 4 - 16 × 16
        x = self.down3(x)
        x = self.res4_1(x, t)
        x = self.res4_2(x, t)
        x = self.attn4(x)
        skip4 = x

        # BOTTLENECK 8 × 8
        x = self.down4(x)
        x = self.bot1(x, t)
        x = self.bot_attention(x)
        x = self.bot2(x, t)

        # UP Level 4
        x = self.up4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.up_res4_1(x,t)
        x = self.up_res4_2(x, t)
        x = self.up_attn4(x)

        # UP Level 3
        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_res3_1(x, t)
        x = self.up_res3_2(x, t)

        # UP Level 2
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_res2_1(x, t)
        x = self.up_res2_2(x, t)

        # UP Level 1
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_res1_1(x, t)
        x = self.up_res1_2(x, t)

        # Output
        x = self.out_norm(x)
        x = self.out_act(x)

        return self.output(x)
   


