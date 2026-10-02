from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from ddpm.ddpm_functions import (
    get_index_from_list,
    sample_timestep,
    forward_diffusion_sample
)

def tensor_to_numpy_image(img_tensor):
    img = img_tensor.detach().cpu()
    if img.ndim == 3: img = img[0]
    img = torch.clamp(img, -1.0, 1.0)
    img = (img + 1.0) / 2.0
    img = img.numpy()
    img = (img * 255.0).astype(np.uint8)
    return img

''' ---------------------------------------------------------------------------------------------------------------- '''
def save_tensor_as_png(img_tensor, save_path):
    img = tensor_to_numpy_image(img_tensor)
    pil_img = Image.fromarray(img, mode="L")
    pil_img.save(save_path)

''' ---------------------------------------------------------------------------------------------------------------- '''
@torch.no_grad()
def sample_from_noise(
    model,
    n_samples,
    img_size,
    device,
    T,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    posterior_mean_coef1,
    posterior_mean_coef2,
    posterior_variance,
    seed=None,
    batch_size=8
):
    """
    Generate n_samples from pure noise - returns a list of tensors with shape [1, H, W].
    """

    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    generated_images = []
    num_batches = int(np.ceil(n_samples / batch_size))

    for batch_idx in tqdm(range(num_batches), desc="Generating from noise"):
        current_bs = min(batch_size, n_samples - len(generated_images))
        x = torch.randn((current_bs, 1, img_size, img_size), device=device)

        for i in range(T - 1, -1, -1):
            t = torch.full((current_bs,), i, device=device, dtype=torch.long)
            x = sample_timestep(
                x,
                t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                posterior_mean_coef1,
                posterior_mean_coef2,
                model,
                posterior_variance
            )

        x = torch.clamp(x, -1.0, 1.0)

        for j in range(current_bs):
            generated_images.append(x[j].detach().cpu())

    return generated_images

''' ---------------------------------------------------------------------------------------------------------------- '''
def save_real_validation_images(val_loader, out_dir, n_images=2000):
    """
    save the images from the validation loader in a new folder to use them as test set that wasnt seen by the network
    """
    saved = 0
    for batch in tqdm(val_loader, desc="Saving real validation images"):
        x = batch[0]
        for i in range(x.shape[0]):
            save_path = os.path.join(out_dir, f"real_val_{saved:05d}.png")
            save_tensor_as_png(x[i], save_path)
            saved += 1

            if saved >= n_images:
                print(f"Saved {saved} real validation images to: {out_dir}")
                return

    print(f"Saved {saved} real validation images to: {out_dir}")

''' ---------------------------------------------------------------------------------------------------------------- '''
def save_generated_list(images, out_dir, prefix):
    for i, img in enumerate(images):
        save_path = os.path.join(out_dir, f"{prefix}_{i:05d}.png")
        save_tensor_as_png(img, save_path)

''' ---------------------------------------------------------------------------------------------------------------- '''
@torch.no_grad()
def reconstruct_from_xt(
    model,
    x_t,
    start_t,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    posterior_mean_coef1,
    posterior_mean_coef2,
    posterior_variance
):
    """
    Start from x_t and reverse from t=start_t down to 0.
    """
    model.eval()
    x = x_t.clone()
    for i in range(start_t, -1, -1):
        t = torch.full(
            (x.shape[0],),
            i,
            device=x.device,
            dtype=torch.long
        )

        x = sample_timestep(
            x,
            t,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_mean_coef1,
            posterior_mean_coef2,
            model,
            posterior_variance
        )

    x = torch.clamp(x, -1.0, 1.0)
    return x

''' ---------------------------------------------------------------------------------------------------------------- '''
def save_reconstruction_triplet(original, noisy, reconstructed, save_path, timestep):
    """
    Save one figure with: ground truth | noisy | reconstructed
    """
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
    plt.title(f"noisy (t={timestep})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(recon_np, cmap="gray", vmin=0, vmax=255)
    plt.title("reconstructed")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

''' ---------------------------------------------------------------------------------------------------------------- '''
@torch.no_grad()
def save_reconstructions(
    model,
    val_loader,
    out_triplet_dir,
    out_recon_only_dir,
    n_images,
    recon_timestep,
    device,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    posterior_mean_coef1,
    posterior_mean_coef2,
    posterior_variance
):

    saved = 0
    model.eval()

    for batch in tqdm(val_loader, desc="Saving reconstructions"):
        x0_batch = batch[0].to(device)
        current_bs = x0_batch.shape[0]
        t = torch.full((current_bs,), recon_timestep, device=device, dtype=torch.long)

        x_t_batch, _ = forward_diffusion_sample(
            x_0=x0_batch,
            t=t,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            noise=None
        )

        recon_batch = reconstruct_from_xt(
            model=model,
            x_t=x_t_batch,
            start_t=recon_timestep,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2,
            posterior_variance=posterior_variance
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

            save_reconstruction_triplet(
                original=x0_batch[i].cpu(),
                noisy=x_t_batch[i].cpu(),
                reconstructed=recon_batch[i].cpu(),
                save_path=triplet_path,
                timestep=recon_timestep
            )

            save_tensor_as_png(
                recon_batch[i].cpu(),
                recon_only_path
            )

            saved += 1

            if saved >= n_images:
                print(f"Saved {saved} reconstructions to: {out_triplet_dir}")
                print(f"Saved {saved} reconstructed-only images to: {out_recon_only_dir}")
                return

    print(f"Saved {saved} reconstructions to: {out_triplet_dir}")
    print(f"Saved {saved} reconstructed-only images to: {out_recon_only_dir}")

''' ---------------------------------------------------------------------------------------------------------------- '''