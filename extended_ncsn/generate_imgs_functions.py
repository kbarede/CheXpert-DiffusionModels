import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def tensor_to_numpy_image(tensor):
    """
    Converts a tensor in [-1, 1] to image in [0, 255].
    """
    if tensor.ndim == 3:tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().float()
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor * 255.0

    return tensor.numpy().astype(np.uint8)

def save_tensor_as_png(tensor, save_path):
    """
    Save a tensor image in [-1,1] as grayscale PNG.
    """
    img_np = tensor_to_numpy_image(tensor)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


@torch.no_grad()
def reconstruct_from_noisy_ncsn(model, x_noisy, sigmas, start_sigma_index, step_lr=2e-5, n_steps_each=50, adaptive_steps=True):
    """
    Start from a perturbed image x_noisy and reconstruct it by running
    annealed Langevin dynamics from start_sigma_index down to sigma_min.
    """
    model.eval()
    x = x_noisy.clone()
    sigma_min = sigmas[-1]
    for sigma_index in range(start_sigma_index, len(sigmas)):
        sigma = sigmas[sigma_index]
        labels = torch.full(
            (x.shape[0],),
            sigma_index,
            device=x.device,
            dtype=torch.long
        )

        step_size = step_lr * (sigma / sigma_min) ** 2

        if adaptive_steps:
            if sigma.item() >= 0.6: current_steps = 100
            elif sigma.item() >= 0.2: current_steps = 80
            elif sigma.item() >= 0.05: current_steps = 40
            else: current_steps = 10
        else: current_steps = n_steps_each

        for _ in range(current_steps):
            scaled_score = model(x, labels)
            score = scaled_score / sigma
            langevin_noise = (torch.randn_like(x) * torch.sqrt(2.0 * step_size))
            x = x + step_size * score + langevin_noise

    # final denoising step
    final_labels = torch.full(
        (x.shape[0],),
        len(sigmas) - 1,
        device=x.device,
        dtype=torch.long
    )

    final_scaled_score = model(x, final_labels)
    final_score = final_scaled_score / sigma_min
    x = x + (sigma_min ** 2) * final_score
    x = torch.clamp(x, -1.0, 1.0)
    return x

def save_reconstruction_triplet_ncsn(original, noisy, reconstructed, save_path, sigma_value):
    orig_np = tensor_to_numpy_image(original)
    noisy_np = tensor_to_numpy_image(noisy)
    recon_np = tensor_to_numpy_image(reconstructed)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(orig_np, cmap="gray", vmin=0, vmax=255)
    plt.title("ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_np, cmap="gray", vmin=0, vmax=255)
    plt.title(f"noisy (σ={sigma_value:.3f})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(recon_np, cmap="gray", vmin=0, vmax=255)
    plt.title("reconstructed")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def save_ncsn_reconstructions(
    model,
    val_loader,
    out_triplet_dir,
    out_recon_only_dir,
    n_images,
    sigmas,
    recon_sigma_index,
    device,
    step_lr=2e-5,
    n_steps_each=50,
    adaptive_steps=True
):

    os.makedirs(out_triplet_dir, exist_ok=True)
    os.makedirs(out_recon_only_dir, exist_ok=True)

    saved = 0
    model.eval()
    sigma_value = sigmas[recon_sigma_index].item()

    for batch in tqdm(val_loader, desc="Saving NCSN reconstructions"):
        x0_batch = batch[0].to(device)
        current_bs = x0_batch.shape[0]

        sigma = sigmas[recon_sigma_index].view(1, 1, 1, 1).to(device)
        noise = torch.randn_like(x0_batch)
        x_noisy_batch = x0_batch + sigma * noise

        recon_batch = reconstruct_from_noisy_ncsn(
            model=model,
            x_noisy=x_noisy_batch,
            sigmas=sigmas,
            start_sigma_index=recon_sigma_index,
            step_lr=step_lr,
            n_steps_each=n_steps_each,
            adaptive_steps=adaptive_steps
        )

        for i in range(current_bs):
            triplet_path = os.path.join(
                out_triplet_dir,
                f"reconstruction_triplet_{saved:05d}.png"
            )

            recon_only_path = os.path.join(
                out_recon_only_dir,
                f"reconstructed_{saved:05d}.png"
            )

            save_reconstruction_triplet_ncsn(
                original=x0_batch[i].cpu(),
                noisy=x_noisy_batch[i].cpu(),
                reconstructed=recon_batch[i].cpu(),
                save_path=triplet_path,
                sigma_value=sigma_value
            )

            save_tensor_as_png(
                recon_batch[i].cpu(),
                recon_only_path
            )

            saved += 1

            if saved >= n_images:
                print(f"Saved {saved} triplets to: {out_triplet_dir}")
                print(f"Saved {saved} recon-only images to: {out_recon_only_dir}")
                return

    print(f"Saved {saved} triplets to: {out_triplet_dir}")
    print(f"Saved {saved} recon-only images to: {out_recon_only_dir}")