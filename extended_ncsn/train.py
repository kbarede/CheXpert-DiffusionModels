'''
Code modified and adapted from:
    the diffusion model - DDPM
    loading data:        https://gitlab.cir.meduniwien.ac.at:8888/cfuerboeck1/chexpert
'''
from monai.data import DataLoader
from common.load_data import load_chexpert
import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import copy
from extended_ncsn.improved_unet import ImprovedUnet
from extended_ncsn.ncsn_functions import (
    geometric_sigma_schedule,
    get_ncsn_loss,
    sample_plot_ncsn,
    diagnose_ncsn_sigmas,
    update_ema
)

torch.backends.cudnn.enabled = False

''' define input --------------------------------------------------------------------------------------------------- '''
# ===============
# MODEL SETTINGS
# ===============
IMG_SIZE = 128
BATCH_SIZE = 16 #16,4
epochs = 200 #5000, 100
limitChexpertImage = 15000 #15000, 8
learning_rate = 1e-4
device = "cuda"

# saving setting
nEpochs_to_save_img_after = 1 #1
nEpochs_to_print_ncsn_diag = 10 #10
nEpochs_to_validate_after = 5 #10

# NCSN setting
NUM_SIGMAS = 50 #50, 20 
SIGMA_MAX = 2 #2.0 #1.0
SIGMA_MIN = 0.01

# sampling setting
N_STEPS_EACH = 20 #50, 20
STEP_LR = 5e-6 #5e-6 #2e-5
EMA_DECAY = 0.999


data_loc = "/PATH/TO/CHEXPERT_DATASET/" # patht to CheXpert data
target = "No Finding" # target class or None (for healthy vs not healthy) available classes are: "No Finding"	"Enlarged Cardiomediastinum"	"Cardiomegaly"	"Lung Opacity"	"Lung Lesion"	"Edema"	"Consolidation"	"Pneumonia"	"Atelectasis"	"Pneumothorax"	"Pleural Effusion"	"Pleural Other"	"Fracture"	"Support Devices"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# for each training a folder will be created in ur saving path with the batch size, epochs, and starting time.
saveingPath = (
    "/PATH/TO/RESULTS-FOLDER/" # add saving path here
    + "BatchSize" + str(BATCH_SIZE)
    + "_Epochs" + str(epochs)
    + "_" + timestamp
    + "/"
)

''' load data in data loader --------------------------------------------------------------------------------------- '''
train_ds,val_ds = load_chexpert(data_loc,IMG_SIZE,target,limit_n=limitChexpertImage)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

batch = next(iter(train_loader)) 
images = batch[0]

print("\n")
print("------Train loader:------")
print("Number of images:",len(train_ds))
print("Number of batches:",len(train_loader))
print("Total data samples:",len(train_loader.dataset))
print("Configured batch size:", train_loader.batch_size)
print("\n")

print("\n")
print("------Valdidation loader:------")
print("Number of images:",len(val_ds))
print("Number of batches:",len(val_loader))
print("Total data samples:",len(val_loader.dataset))
print("Configured batch size:", val_loader.batch_size)
print("\n")

# check for debug:
print("------Data:------")
print("shape:", images.shape) 
print("min:", images.min().item()) 
print("max:", images.max().item()) 
print("mean:", images.mean().item()) 
print("std:", images.std().item())
print("\n")

if not os.path.exists(saveingPath):
    os.makedirs(saveingPath)

''' define model parameters ------------------------------------------------------------------------------ '''
sigmas = geometric_sigma_schedule(
    sigma_max=SIGMA_MAX,
    sigma_min=SIGMA_MIN,
    num_sigmas=NUM_SIGMAS,
    device=device
)

for i, sigma in enumerate(sigmas):
    print(f"{i:02d}: "f"{sigma.item():.6f}")


model = ImprovedUnet().to(device) 

# =================
# debug:
x = next(iter(train_loader))[0].to(device)

labels = torch.randint(
    0,
    NUM_SIGMAS,
    (x.shape[0],),
    device=device
)

with torch.no_grad():
    y = model(
        x,
        labels
    )

print("input :", x.shape)
print("output:", y.shape)

number_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: "
    f"{number_parameters:,}"
)
# =================

# ====== ema ======
ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad_(False) #=do not calculate gradients for the EMA model.
# ====== ema ======

optimizer = Adam(model.parameters(), lr=learning_rate)
torch.manual_seed(0)
fixed_noise = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=device) * SIGMA_MAX

# number_parameters = sum(
#     p.numel()
#     for p in model.parameters()
# )

# trainable_parameters = sum(
#     p.numel()
#     for p in model.parameters()
#     if p.requires_grad
# )

train_losses = []
val_losses = []
n_val_losses = []



# plot few loaded images to see what images the model is using:
# fig, axes = plt.subplots(2, 4, figsize=(10, 5))
# for i, ax in enumerate(axes.flat): 
#     img = (images[i, 0].cpu() + 1) / 2 
#     ax.imshow(img, cmap='gray') 
#     ax.axis('off')
# plt.tight_layout() 
# plt.savefig(saveingPath + "debug_train_batch.png", dpi=150) 
# plt.close()

''' Evaluation function --------------------------------------------------------------------------------------- '''
@torch.no_grad()
def calculate_validation_loss():
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader: 
        x = batch[0].to(device)
        loss = get_ncsn_loss(model, x, sigmas)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches

''' Model training --------------------------------------------------------------------------------------- '''
global_step = 0

for epoch in tqdm(range(epochs)):
    epoch_loss = 0.0
    num_batches = 0
    model.train()

    for step, batch in enumerate(train_loader):
      optimizer.zero_grad()
      x = batch[0].to(device, non_blocking=True)
      loss = get_ncsn_loss(model, x, sigmas)
      loss.backward()
      # if network got larage gradients:
      #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      update_ema(ema_model, model, decay=EMA_DECAY)
      global_step += 1      
      epoch_loss += loss.item()
      num_batches += 1

    avg_loss = epoch_loss/num_batches 
    train_losses.append(avg_loss)
    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Step {global_step} | "
        f"Avg Loss: {avg_loss:.6f}"
    ) 

    if (epoch + 1) % nEpochs_to_save_img_after == 0:
        print(
        f"\n===== EPOCH {epoch + 1} "
        f"| GLOBAL STEP {global_step} ====="
        )
        sample_plot_ncsn( 
            saveingPath + f"figure_{epoch+1:04d}.png",
            model=ema_model, #model or ema_model
            fixed_noise=fixed_noise,
            sigmas=sigmas,
            n_steps_each=N_STEPS_EACH,
            step_lr=STEP_LR,
            adaptive_steps=False 
        )
    
    if (epoch +1) % nEpochs_to_print_ncsn_diag ==0:
        diagnose_ncsn_sigmas(model, train_loader, sigmas, device)
        
    # Save model, loss, chechpoint
    pd.DataFrame({'loss': train_losses}).to_csv( 
        saveingPath + 'training_loss.csv', index=False)
    torch.save(model.state_dict(), saveingPath + "model_interim.pth") 
    torch.save(model.state_dict(),saveingPath + "model")
    torch.save(ema_model.state_dict(), saveingPath + "ema_model.pth")
    checkpoint_path = saveingPath + "checkpoint.pth"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'num_sigmas': NUM_SIGMAS,
            'sigma_max': SIGMA_MAX,
            'sigma_min': SIGMA_MIN,
            'n_steps_each': N_STEPS_EACH,
            'step_lr': STEP_LR,
            'ema_decay': EMA_DECAY
        }, checkpoint_path)

    #plot loss 
    n_epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        n_epochs,
        train_losses,
        marker="o",
        linewidth=2,
        markersize=4,
        label="Training loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
    f"NCSN Training Loss\n"
    f"Batch Size: {BATCH_SIZE} | Learning Rate: {learning_rate:.1e}"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(saveingPath + "training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Model evaluation 
    if ((epoch + 1)% nEpochs_to_validate_after == 0):
        val_loss = (calculate_validation_loss())
        val_losses.append(val_loss)
        pd.DataFrame({'loss': val_losses}).to_csv( 
            saveingPath + 'val_loss.csv', index=False)
        
        n_val_losses.append((epoch + 1))

        print(n_val_losses)
        print(val_losses)

        plt.figure(figsize=(10, 6))
        plt.plot(
            n_val_losses,
            val_losses,
            marker="o",
            linewidth=2,
            markersize=4,
            label="Validation loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
        f"Diffusion Model Validation Loss\n"
        f"Batch Size: {BATCH_SIZE} | Learning Rate: {learning_rate:.1e}"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(saveingPath + "validation_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

