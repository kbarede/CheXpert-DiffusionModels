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
class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1
        time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear( time_emb_dim * 4, time_emb_dim)
        )

        self.conv0 = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.res1_1 = ResBlock(64, 64, time_emb_dim)
        self.res1_2 = ResBlock(64, 64, time_emb_dim)
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.res2_1 = ResBlock(128, 128, time_emb_dim)
        self.res2_2 = ResBlock(128, 128, time_emb_dim)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.res3_1 = ResBlock(256, 256, time_emb_dim)
        self.res3_2 = ResBlock(256, 256, time_emb_dim)
        self.down3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # BOTTLENECK
        self.bot1 = ResBlock(512, 512, time_emb_dim)
        self.bot2 = ResBlock(512, 512, time_emb_dim)

        # UP
        self.up3 = nn.ConvTranspose2d(512,256, kernel_size=4, stride=2,padding=1)
        self.up_res3_1 = ResBlock(512, 256, time_emb_dim)
        self.up_res3_2 = ResBlock(256, 256,time_emb_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up_res2_1 = ResBlock(256, 128, time_emb_dim)
        self.up_res2_2 = ResBlock(128, 128, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(128,64, kernel_size=4, stride=2, padding=1)
        self.up_res1_1 = ResBlock(128, 64, time_emb_dim)
        self.up_res1_2 = ResBlock(64, 64, time_emb_dim)

        # Output
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_act = nn.SiLU()
        self.output = nn.Conv2d(64, 1,kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        x = self.res1_1(x, t)
        x = self.res1_2(x, t)

        skip1 = x
        x = self.down1(x)
        x = self.res2_1(x, t)
        x = self.res2_2(x, t)

        skip2 = x
        x = self.down2(x)
        x = self.res3_1(x, t)
        x = self.res3_2(x, t)

        skip3 = x
        x = self.down3(x)

        # BOTTLENECK
        x = self.bot1(x, t)
        x = self.bot2(x, t)

        # UP
        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_res3_1(x, t)
        x = self.up_res3_2(x, t)
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_res2_1(x, t)
        x = self.up_res2_2(x, t)
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_res1_1(x, t)
        x = self.up_res1_2(x, t)

        # OUTPUT
        x = self.out_norm(x)
        x = self.out_act(x)

        return self.output(x)
   


