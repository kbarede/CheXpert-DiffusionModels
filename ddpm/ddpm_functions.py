import torch
from torch import nn
import math
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

''' ---------------------------------------------------------------------------------------------------------------- '''
def show_tensor_image(image):
    if image.ndim == 4: image = image[0]
    image = image.detach().cpu()
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1.0) / 2.0
    plt.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
    

''' ---------------------------------------------------------------------------------------------------------------- '''
def forward_diffusion_sample(x_0, t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, device="cuda"):
    """
    Takes an image and a timestep as input and returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

''' ---------------------------------------------------------------------------------------------------------------- '''


def get_loss(model, device, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    x_noisy, noise = forward_diffusion_sample( x_0, 
                                              t, 
                                              sqrt_alphas_cumprod, 
                                              sqrt_one_minus_alphas_cumprod, 
                                              device 
                                            ) 
    noise_pred = model(x_noisy, t)

    # Debug every ~1% of batches 
    if torch.rand(1).item() < 0.05: 
        with torch.no_grad(): 
            corr = torch.corrcoef( 
                torch.stack([ 
                    noise.flatten(), 
                    noise_pred.flatten() 
                ]) 
            )[0, 1] 
            print("noise mean:", noise.mean().item()) 
            print("noise std:", noise.std().item()) 
            print("pred mean:", noise_pred.mean().item()) 
            print("pred std:", noise_pred.std().item()) 
            print("corr:", corr.item())

    return F.mse_loss(noise_pred, noise)
''' ---------------------------------------------------------------------------------------------------------------- '''

@torch.no_grad()
def sample_timestep(
    x,
    t,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    posterior_mean_coef1,
    posterior_mean_coef2,
    model,
    posterior_variance
):
    pred_noise = model(x, t)
    sqrt_alpha_bar_t = get_index_from_list(sqrt_alphas_cumprod, t,x.shape)
    sqrt_one_minus_alpha_bar_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x.shape
    )

    pred_x0 = (
        x
        - sqrt_one_minus_alpha_bar_t * pred_noise
    ) / (sqrt_alpha_bar_t + 1e-8)

    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

    coef1_t = get_index_from_list(posterior_mean_coef1, t,x.shape)
    coef2_t = get_index_from_list(posterior_mean_coef2,t,x.shape)
    model_mean = (coef1_t * pred_x0 + coef2_t * x)

    if t[0].item() == 0:
        return model_mean

    posterior_variance_t = get_index_from_list(
        posterior_variance,
        t,
        x.shape
    )

    noise = torch.randn_like(x)

    return (
        model_mean
        + torch.sqrt(
            torch.clamp(posterior_variance_t, min=1e-20)
        ) * noise
    )


@torch.no_grad()    
def sample_plot_image( 
    path, 
    fixed_noise, 
    device, 
    T, 
    betas, 
    sqrt_one_minus_alphas_cumprod, 
    model, 
    posterior_variance,
    sqrt_alphas_cumprod,
    posterior_mean_coef1,
    posterior_mean_coef2
    ):
     
    model.eval() 
    img = fixed_noise.clone()

    vis_steps = [T-1, int(T*0.8), int(T*0.6), int(T*0.4), int(T*0.2), int(T*0.1), 0] 
    vis_steps = sorted(list(set(vis_steps)), reverse=True)

    saved_images = {}

    for i in range(T-1, -1, -1): 
        t = torch.full((img.shape[0],), i, device=device, dtype=torch.long) 
        img = sample_timestep(
            img,
            t,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_mean_coef1,
            posterior_mean_coef2,
            model,
            posterior_variance
            )

        # DEBUG: check image BEFORE clamping
        # if i in [T-1, T-2, T-3, 800, 600, 400, 200, 100, 50, 10, 5, 0]:
        #     print(
        #     f"t={i}: "
        #     f"min={img.min().item():.3f}, "
        #     f"max={img.max().item():.3f}, "
        #     f"mean={img.mean().item():.3f}, "
        #     f"std={img.std().item():.3f}"
        #     )

        if i in vis_steps: 
            saved_images[i] = img.detach().cpu().clone() 
            

    img = torch.clamp(img, -1.0, 1.0) 
    saved_images[0] = img.detach().cpu().clone()

    plt.figure(figsize=(18, 3)) 

    for idx, step in enumerate(vis_steps): 
        plt.subplot(1, len(vis_steps), idx + 1) 
        show_tensor_image(saved_images[step]) 
        
        if step == T - 1: plt.title('noise') 
        elif step == 0: plt.title('final')
        else: plt.title(f't={step}') 
            
    plt.tight_layout() 
    plt.savefig(path, dpi=150) 
    plt.close()
