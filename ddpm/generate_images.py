'''
Generate 1000 images using DDPM model.

'''
import torch
from torch.optim import Adam
import torch.nn.functional as F
#from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from common.model import SimpleUnet
from ddpm.ddpm_functions import (
    linear_beta_schedule,
    get_index_from_list,
    sample_timestep
)
from ddpm.generation_functions import (
    save_real_validation_images,
    sample_from_noise,  
    save_generated_list,
    save_reconstructions  
)

torch.backends.cudnn.enabled = False

''' ADJUST PATHS --------------------------------------------------------------------------------------------------- '''

# results_path: add your results folder that contain your results. New folder will be created for generated images.
results_path = "/PATH/TO/RESULTS-FOLDER/" 
save_root= os.path.join(results_path, "generated_images/")
gen_1000_dir = os.path.join(save_root, "generated_from_pure_noise")
os.makedirs(gen_1000_dir, exist_ok=True)

# Adjust this path to your model checkpoint
model_path = os.path.join(results_path, "PATH/TO/CHECKPOINT.pth") 


''' define input --------------------------------------------------------------------------------------------------- '''
# MODEL SETTINGS
IMG_SIZE = 128
BATCH_SIZE = 16
T = 1000 
limitChexpertImage = 15000
device = "cuda"

# N images
N_REAL_VAL_IMAGES = 1000
N_GENERATED_1000 = 1000
N_RECONSTRUCTIONS = 100
RECON_TIMESTEP = T - 1

''' define diffusion model parameters - for reco and generating ------------------------------------ '''
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
posterior_mean_coef2 = ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

#load model:
model = SimpleUnet().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded from:", model_path)


''' generate 1000 images from pure noise ------------------------------------------------------------------------------ '''
generated_1000 = sample_from_noise(
    model=model,
    n_samples=1000,
    img_size=IMG_SIZE,
    device=device,
    T=T,
    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
    posterior_mean_coef1=posterior_mean_coef1,
    posterior_mean_coef2=posterior_mean_coef2,
    posterior_variance=posterior_variance,
    seed=0,
    batch_size=8
)

save_generated_list(generated_1000, gen_1000_dir, prefix="ddpm_gen")
print("Saved 1000 generated images to:", gen_1000_dir)
print("\nDone.")
