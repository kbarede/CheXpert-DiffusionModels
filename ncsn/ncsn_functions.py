import torch
from torch import nn
import math
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math


def get_ncsn_loss(model, x, sigmas):
    
    #Use each sigma for all images and average over the loss!
    #https://github.com/Glanceyes/ML-Paper-Review/blob/main/ComputerVision/Diffusion/NCSN/NCSN.ipynb

    # loss = 0.0
    # for sigma in sigma_t:
    #     sigma = sigma.repeat(x.shape[0]).to(device)
    #     x_perturbed = x + sigma[:, None, None, None] * torch.randn_like(x)
    #     score = model(x_perturbed, sigma)
    #     score_predicted = (score * sigma[:, None, None, None]).view(x.shape[0], -1)
    #     target = (-(x_perturbed - x) / (sigma[:, None, None, None])).view(x.shape[0], -1)
    #     loss += 1. / 2. * ((score_predicted - target) ** 2.).sum(dim=-1).mean(dim=0)
    # loss = loss / sigma_t.size(0)
    # # print(loss)
    # return loss

    """
    The network predicts the SCALED score: scaled_score = sigma * score
    Therefore the target is simply: -epsilon
    The actual score used during sampling is: score = scaled_score / sigma
    """

    batch_size = x.shape[0]
    labels = torch.randint(low=0, high=len(sigmas), size=(batch_size,), device=x.device).long()
    used_sigmas = sigmas[labels].view(batch_size, 1, 1, 1) # sigma for every image. Shape: [B] = [B,1,1,1]
    noise = torch.randn_like(x) # Gaussian noise
    x_noisy = (x + used_sigmas * noise)
    scaled_score_pred = model(x_noisy,labels) # sigma * score
    target = -noise 

    #score_pred = model(x_noisy, labels)
    # maybe check another loss functions Equivalent to:
    # sigma^2 * || score_pred - (-noise / sigma) ||^2
    loss = 0.5 * F.mse_loss(scaled_score_pred , target)

    # Debug
    # if torch.rand(1).item() < 0.002:
    #     with torch.no_grad():
    #         corr = torch.corrcoef(
    #             torch.stack([
    #                 target.flatten(),
    #                 scaled_score_pred.flatten()
    #             ])
    #         )[0, 1]            
    #         print("\n===== NCSN DEBUG =====")
    #         print("sigma mean:", used_sigmas.mean().item())
    #         print("noise mean:", noise.mean().item())
    #         print("noise std:", noise.std().item())
    #         print("pred mean:", scaled_score_pred.mean().item())
    #         print("pred std:", scaled_score_pred.std().item())
    #         print("corr:", corr.item())
    # return loss

''' ---------------------------------------------------------------------------------------------------------------- '''
def geometric_sigma_schedule(sigma_max=1.0, sigma_min=0.01, num_sigmas=20, device="cuda"):
    sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), num_sigmas, device=device))
    #print("sigmas:", sigmas)
    return sigmas

''' ---------------------------------------------------------------------------------------------------------------- '''
# annealed Langevin dynamics
#https://github.com/ermongroup/ncsn/blob/master/runners/anneal_runner.py#L186
@torch.no_grad()
def annealed_langevin_dynamics(model, initial_noise, sigmas, n_steps_each=20, step_lr=5e-6, save_intermediate=False, adaptive_steps=False):
    """
    Generate images from noise using annealed Langevin dynamics.
    For every sigma: x <- x + alpha * score(x,sigma) + sqrt(2*alpha) * random_noise
    where: alpha = step_lr * (sigma / sigma_min)^2
    """
    model.eval()
    x = initial_noise.clone()
    sigma_min = sigmas[-1]
    intermediate_images = {}
    for sigma_index, sigma in enumerate(sigmas): # Move from largest sigma to smallest sigma
        labels = torch.full(
            (x.shape[0],),
            sigma_index,
            device=x.device,
            dtype=torch.long
        )
        step_size = (step_lr * (sigma / sigma_min) ** 2)

        # Langevin steps for this sigma - predict sigma * score
        #for step in range(n_steps_each):
            
            # scaled_score = model(x, labels)
            # score = (scaled_score/ sigma)
            # langevin_noise = torch.randn_like(x) *  torch.sqrt(2.0 * step_size)
            # x = (
            #     x
            #     + step_size 
            #     * score
            #     + langevin_noise
            # )
            # #print("sigma_index: {}, step_size: {}, mean {}, max {}"
            # #      .format(sigma_index, step_size, score.abs().mean(), score.abs().max()))

        if adaptive_steps: #sampler used after training is mmore expensive than when training
            if sigma.item() >= 0.6: current_steps = 100   
            elif sigma.item() >= 0.2: current_steps = 80  
            elif sigma.item() >= 0.05: current_steps = 40  
            else: current_steps = 10  
        else: #cheap sample for training
            current_steps = n_steps_each

        for step in range(current_steps):
            scaled_score = model(x, labels)
            score = scaled_score / sigma
            langevin_noise = (torch.randn_like(x) * torch.sqrt(2.0 * step_size))
            x = (x + step_size * score + langevin_noise)

        if save_intermediate:
            intermediate_images[sigma_index] = (x.detach().cpu().clone())

    # Final denoising step x0 approximately: x + sigma_min^2 * score
    final_labels = torch.full((x.shape[0],), len(sigmas) - 1, device=x.device, dtype=torch.long)
    final_scaled_score = model(x,final_labels)
    final_score = (final_scaled_score / sigma_min)
    x = (x + sigma_min ** 2 * final_score)
    # X-rays normalized to [-1,1]
    x = torch.clamp(x, -1.0,1.0)
   
    return x, intermediate_images

''' ---------------------------------------------------------------------------------------------------------------- '''
def show_ncsn_image(image):
    if image.ndim == 4: image = image[0]
    image = (image.detach().cpu())
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1.0) / 2.0
    plt.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)

''' ---------------------------------------------------------------------------------------------------------------- '''
@torch.no_grad()
def sample_plot_ncsn(path, model, fixed_noise, sigmas, n_steps_each=20, step_lr=5e-6, adaptive_steps=False):
    model.eval()
    generated, intermediate = (
        annealed_langevin_dynamics(
            model=model,
            initial_noise=fixed_noise,
            sigmas=sigmas,
            n_steps_each=n_steps_each,
            step_lr=step_lr,
            save_intermediate=True,
            adaptive_steps=adaptive_steps
        )
    )

    # Select sigma levels to visualize
    n = len(sigmas)
    selected_indices = [
        0,
        int(n * 0.2),
        int(n * 0.4),
        int(n * 0.6),
        int(n * 0.8),
        n - 1
    ]

    selected_indices = sorted(list(set(selected_indices)))
    plt.figure(figsize=(18, 3))

    # Initial noise
    total_panels = (len(selected_indices) + 2)
    plt.subplot(1,total_panels,1)
    show_ncsn_image(fixed_noise)
    plt.title("noise")

    # Intermediate sigma levels
    for plot_index, sigma_index in enumerate(selected_indices):
        plt.subplot(1,total_panels, plot_index + 2)
        show_ncsn_image(intermediate[sigma_index])
        plt.title(f"σ={sigmas[sigma_index].item():.3f}")

    # Final generated image
    plt.subplot(1,total_panels,total_panels)
    show_ncsn_image(generated)
    plt.title("final")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    model.train()

''' ---------------------------------------------------------------------------------------------------------------- '''

''' Debug/Diagnos function --------------------------------------------------------------------------------------- '''
@torch.no_grad()
def diagnose_ncsn_sigmas(model, train_loader, sigmas, device):
    model.eval()
    batch = next(iter(train_loader))
    x = batch[0].to(device)
    noise = torch.randn_like(x)
    test_indices = [0, 2, 5, 8, 11, 14, 17, len(sigmas) - 1]

    print("\n========== NCSN SIGMA DIAGNOSTIC ==========")
    for idx in test_indices:
        sigma = sigmas[idx]
        x_noisy = x + sigma * noise
        labels = torch.full((x.shape[0],), idx, device=device, dtype=torch.long)
        pred_scaled = model(x_noisy, labels) # sigma * predicted_score should approximate -noise. we want to predict scaled score.
        target_scaled = -noise
        mse = F.mse_loss(pred_scaled, target_scaled).item()
        corr = torch.corrcoef(
            torch.stack([
                target_scaled.flatten(),
                pred_scaled.flatten()
            ])
        )[0, 1].item()

        print(
            f"sigma={sigma.item():.5f} | "
            f"MSE={mse:.5f} | "
            f"corr={corr:.4f} | "
            f"pred_std={pred_scaled.std().item():.4f}"
        )
    model.train()

''' ---------------------------------------------------------------------------------------------------------------- '''
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    #θEMAnew​=dθEMAold​+(1−d)θmodel​, where d=decy and θ=weight
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(
            model_param.data,
            alpha=1.0 - decay
        )

