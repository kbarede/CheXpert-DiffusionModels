'''
Code modified and adapted from:
    the diffusion model: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=i7AZkYjKgQTm
    loading data:        https://gitlab.cir.meduniwien.ac.at:8888/cfuerboeck1/chexpert
'''
from monai.data import DataLoader
from common.load_data import load_chexpert
from common.model import SimpleUnet
from ddpm.ddpm_functions import (
    linear_beta_schedule,
    get_loss,
    sample_plot_image,
    forward_diffusion_sample, 
    get_index_from_list
)
import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.enabled = False

''' define input --------------------------------------------------------------------------------------------------- '''
# ===============
# MODEL SETTINGS
# ===============
IMG_SIZE = 128
BATCH_SIZE = 16
epochs = 100 #3000
max_training_steps = 10000
T = 1000 
limitChexpertImage = 13000 
learning_rate = 1e-4
device = "cuda"

data_loc = "/PATH/TO/CHEXPERT_DATASET/" # patht to CheXpert data
target = "No Finding" # target class or None (for healthy vs not healthy) available classes are: "No Finding"	"Enlarged Cardiomediastinum"	"Cardiomegaly"	"Lung Opacity"	"Lung Lesion"	"Edema"	"Consolidation"	"Pneumonia"	"Atelectasis"	"Pneumothorax"	"Pleural Effusion"	"Pleural Other"	"Fracture"	"Support Devices"

# for each training a folder will be created in ur saving path with the batch size, epochs, and starting time.
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
batch = next(iter(train_loader)) 
images = batch[0]

# print("Number of images:",len(train_ds))
# print("Number of batches:",len(train_loader))
# print("Total data samples:",len(train_loader.dataset))
# print("Configured batch size:", train_loader.batch_size) 

# # check for debug:
# print("shape:", images.shape) 
# print("min:", images.min().item()) 
# print("max:", images.max().item()) 
# print("mean:", images.mean().item()) 
# print("std:", images.std().item())

''' define diffusion model parameters ------------------------------------------------------------------------------ '''
# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
#sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
posterior_mean_coef2 = ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

# # check for debug:
# print("beta first:", betas[0].item())
# print("beta last:", betas[-1].item())
# print("alpha_bar first:", alphas_cumprod[0].item())
# print("alpha_bar final:", alphas_cumprod[-1].item())
# print("sqrt alpha final:", sqrt_alphas_cumprod[-1].item())
# print("sqrt 1-alpha final:", sqrt_one_minus_alphas_cumprod[-1].item())

model = SimpleUnet().to(device) 
optimizer = Adam(model.parameters(), lr=learning_rate) 
torch.manual_seed(0)
fixed_noise = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=device)

# test model for debug:
test_x = torch.randn(4, 1, IMG_SIZE, IMG_SIZE, device=device)
test_t = torch.randint( 0, T, (4,), device=device).long()

with torch.no_grad():
    test_out = model(
        test_x,
        test_t
    )

print("TEST INPUT:", test_x.shape)
print("TEST OUTPUT:", test_out.shape)
assert test_out.shape == test_x.shape
print("Model shape test passed!")

if not os.path.exists(saveingPath):
    os.makedirs(saveingPath)

# plot few loaded images to see what images the model is using:
# fig, axes = plt.subplots(2, 4, figsize=(10, 5))
# for i, ax in enumerate(axes.flat): 
#     img = (images[i, 0].cpu() + 1) / 2 
#     ax.imshow(img, cmap='gray') 
#     ax.axis('off')
# plt.tight_layout() 
# plt.savefig(saveingPath + "debug_train_batch.png", dpi=150) 
# plt.close()

''' Model training --------------------------------------------------------------------------------------- '''

train_losses = []

for epoch in tqdm(range(epochs)):
    epoch_loss = 0.0
    num_batches = 0
    model.train()

    for step, batch in enumerate(train_loader):
      optimizer.zero_grad()
      x = batch[0].to(device)
      t = torch.randint(0, T, (x.shape[0],), device=device).long()
      loss = get_loss(
          model,
          device, 
          x, 
          t,
          sqrt_alphas_cumprod,
          sqrt_one_minus_alphas_cumprod
          )
      
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      num_batches += 1

    avg_loss = epoch_loss/num_batches 
    train_losses.append(avg_loss)  
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")

    if (epoch + 1) % 1 == 0:
        sample_plot_image( saveingPath + f"figure_{epoch+1:04d}.png",
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
                    )
        
    # Save model, loss, chechpoint
    pd.DataFrame({'loss': train_losses}).to_csv( 
        saveingPath + 'training_loss.csv', index=False )
    torch.save(model.state_dict(), saveingPath + "model_interim.pth") 
    torch.save(model.state_dict(),saveingPath + "model")
    checkpoint_path = saveingPath + "checkpoint.pth"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
    f"Diffusion Model Training Loss\n"
    f"Batch Size: {BATCH_SIZE} | Learning Rate: {learning_rate:.1e}"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(saveingPath + "training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
''' Reconstruction test --------------------------------------------------------------------------------------- '''

batch = next(iter(train_loader)) 
x0 = batch[0][:1].to(device) 
t = torch.tensor([int(T*0.2)], device=device) # noise chooes from timestep 200 - for strict test use T-1
x_noisy, noise = forward_diffusion_sample(x0, t,  sqrt_alphas_cumprod,  sqrt_one_minus_alphas_cumprod, device)

model.train()
with torch.no_grad():
    pred_train = model(x_noisy, t)
corr_train = torch.corrcoef(torch.stack([ noise.flatten(), pred_train.flatten()]))[0, 1]
mse_train = F.mse_loss(pred_train, noise)

model.eval() 
with torch.no_grad(): 
    pred_eval = model(x_noisy, t)
corr_eval = torch.corrcoef(torch.stack([ noise.flatten(), pred_eval.flatten()]))[0, 1] 
mse_eval = F.mse_loss(pred_eval, noise)

sqrt_alpha_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape ) 
sqrt_one_minus_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape ) 
x0_pred = (x_noisy - sqrt_one_minus_t * pred_train) / sqrt_alpha_t 
fig, ax = plt.subplots(1, 3, figsize=(9, 3)) 

ax[0].imshow(((x0[0,0].cpu() + 1) / 2), cmap='gray') 
ax[0].set_title('original') 
ax[1].imshow(((x_noisy[0,0].cpu() + 1) / 2), cmap='gray') 
ax[1].set_title('noisy') 
ax[2].imshow(((x0_pred[0,0].cpu() + 1) / 2), cmap='gray') 
ax[2].set_title('reconstructed') 

for a in ax: 
    a.axis('off') 

plt.tight_layout() 
plt.savefig(saveingPath + 'reconstruction_test.png', dpi=150) 
plt.close() 


print("\n====== BATCHNORM TEST ======")
print(
    f"TRAIN mode: "
    f"MSE={mse_train.item():.6f}, "
    f"corr={corr_train.item():.6f}, "
    f"std={pred_train.std().item():.6f}"
)

print(
    f"EVAL mode:  "
    f"MSE={mse_eval.item():.6f}, "
    f"corr={corr_eval.item():.6f}, "
    f"std={pred_eval.std().item():.6f}"  )