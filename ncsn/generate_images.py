'''
Generate 1000 images using NCSN model.

'''
import torch
import os
import matplotlib.pyplot as plt
from common.model import SimpleUnet
from ncsn.ncsn_functions import (
    geometric_sigma_schedule,
    annealed_langevin_dynamics,
    update_ema,
    diagnose_ncsn_sigmas,
    get_ncsn_loss,
    sample_plot_ncsn
)

from ncsn.generate_imgs_functions import (
    save_ncsn_reconstructions
)

torch.backends.cudnn.enabled = False

''' ADJUST PATHs --------------------------------------------------------------------------------------------------- '''
# results_path: add your results folder that contain your results. New folder will be created for generated images.
results_path ="/PATH/TO/RESULTS-FOLDER/"
save_root= os.path.join(results_path, "generated_images/")
norm_gen_1000_dir = os.path.join(save_root, "normal_model/")
ema_gen_1000_dir = os.path.join(save_root, "ema/")

# Adjust checkpoint_path path to your model checkpoint
checkpoint_path = os.path.join(results_path,"PATH/TO/CHECKPOINT.pth")

''' define input --------------------------------------------------------------------------------------------------- '''
# ===============
# MODEL SETTINGS
# ===============
IMG_SIZE = 128
BATCH_SIZE = 16
GEN_BATCH_SIZE = 8
limitChexpertImage = 15000 #15000, 8
device = "cuda"

#Num Images
N_GENERATED_IMGs = 1000
N_RECONSTRUCTIONS = 100

# NCSN setting
NUM_SIGMAS = 50 #50, 20 
SIGMA_MAX = 2 #2.0 #1.0
SIGMA_MIN = 0.01

# sampling setting
N_STEPS_EACH = 20 #50, 20
STEP_LR = 5e-6 #5e-6 #2e-5
EMA_DECAY = 0.999 
adaptive_steps = True # unlike training, use expensive sampling

''' define model parameters ------------------------------------------------------------------------------ '''
sigmas = geometric_sigma_schedule(
    sigma_max=SIGMA_MAX,
    sigma_min=SIGMA_MIN,
    num_sigmas=NUM_SIGMAS,
    device=device
)

for i, sigma in enumerate(sigmas):
    print(f"{i:02d}: "f"{sigma.item():.6f}")

''' load model ------------------------------------------------------------------------------ '''

model = SimpleUnet().to(device)
ema_model = SimpleUnet().to(device)  
checkpoint = torch.load(checkpoint_path, map_location=device)
if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    print("Loaded checkpoint epoch:", checkpoint.get("epoch", "unknown"))
else:
    model.load_state_dict(checkpoint)

model.eval()
ema_model.eval()
generated_count = 0

''' generate 1000 images from pure noise ------------------------------------------------------------------------------ '''
with torch.no_grad():
    while generated_count < N_GENERATED_IMGs:
        current_batch_size = min(GEN_BATCH_SIZE, N_GENERATED_IMGs - generated_count)

        initial_noise = (
            torch.randn(
                current_batch_size,
                1,
                IMG_SIZE,
                IMG_SIZE,
                device=device
            )
            * SIGMA_MAX
        )

        ema_generated, _ = annealed_langevin_dynamics(
            model=ema_model,
            initial_noise=initial_noise,
            sigmas=sigmas,
            n_steps_each=N_STEPS_EACH,
            step_lr=STEP_LR,
            save_intermediate=False,
            adaptive_steps=True
        )

        generated, _ = annealed_langevin_dynamics(
            model=model,
            initial_noise=initial_noise,
            sigmas=sigmas,
            n_steps_each=N_STEPS_EACH,
            step_lr=STEP_LR,
            save_intermediate=False,
            adaptive_steps=True
        )

        for i in range(current_batch_size):
            ema_img = ema_generated[i, 0].detach().cpu()
            img = generated[i, 0].detach().cpu()

            # [-1, 1] -> [0, 1]
            ema_img = torch.clamp(ema_img, -1.0, 1.0)
            ema_img = (ema_img + 1.0) / 2.0

            img = torch.clamp(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0

            plt.imsave(
                ema_gen_1000_dir
                + f"ncsn_{generated_count:05d}.png",
                ema_img.numpy(),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            plt.imsave(
                norm_gen_1000_dir
                + f"ncsn_{generated_count:05d}.png",
                img.numpy(),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            generated_count += 1
        print(
            f"Generated "
            f"{generated_count}/{N_GENERATED_IMGs}"
        )


print("Finished generating 1000 NCSN images.")


