import os
os.environ['NUMEXPR_MAX_THREADS'] = '4'

import pandas as pd
from monai.data import ImageDataset
#from monai.transforms import Compose, Resize, ResizeWithPadOrCrop, ToTensor, Lambda
from monai.transforms import Compose, AddChannel,EnsureChannelFirst, Resize, ScaleIntensity, Lambda
from monai.transforms.transform import Transform
import torch


# ====================================================================================================================================

class IntensityScale(Transform):
    """
    Apply intensity scaling to the whole numpy array.
    Scaling images to the range of [a,b].
    """

    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        img = img * (self.max_val - self.min_val) + self.min_val

        return img

class Unsqueeze(Transform):
    """
    Unsqeeze image tensor at given dimension.
    """
    def __init__(self,dim=0):
        self.dim = dim

    def __call__(self, img):
        return torch.unsqueeze(img,self.dim)


# ====================================================================================================================================


def load_chexpert(data_loc,img_size,target=None,limit_n=0):

    # read csv with all information about the dataset
    df_overview = pd.read_csv(data_loc+"CheXpert-v1.0-small/train.csv")
    df_overview['img_path'] = [f'{data_loc}/{pat}' for pat in df_overview.Path]

    if target is not None:
        # filter out for target class
        df_overview = df_overview[df_overview[target].notnull()]
        df_overview = df_overview[df_overview[target] != 0]
    else: target = "No Finding"

    df_overview[target] = df_overview[target].apply(lambda x : x if x > 0 else 0)

    # only take Frontal view
    df_overview = df_overview[df_overview["Frontal/Lateral"]=="Frontal"]

    if limit_n>0:
        # take subset of total dataset (for proto-typing)
        df_overview = df_overview.sample(n=limit_n, random_state=0).reset_index(drop=True)  # Seed 0 for reproducibility

    # set train & validation split
    # --> improve this by doing a stratified split
    n_train = int(len(df_overview)*0.866666667)
    df_train = df_overview[:n_train]
    df_val = df_overview[n_train:].reset_index(drop=True)

    # pre-process data: norm values between 0 and 1, rotate & flip, crop/pad to get same size for all
    train_transforms = Compose([
        # ToTensor(),
        # IntensityScale(min_val=0,max_val=1), 
        # Lambda(lambda x: x * 2 - 1),
        # Unsqueeze(), 
        # #Rotate90(k=1, spatial_axes=(1, 0)),
        # #Flip(spatial_axis=1),
        # ResizeWithPadOrCrop(spatial_size=(img_size, img_size))

        # ToTensor(),
        # Unsqueeze(),
        # ResizeWithPadOrCrop(spatial_size=(img_size, img_size)),
        # IntensityScale(min_val=0,max_val=1), 
        # Lambda(lambda x: x * 2 - 1)

        AddChannel(),
        Resize((img_size, img_size)), 
        ScaleIntensity(minv=0.0, maxv=1.0), 
        Lambda(lambda x: x * 2 - 1),
        
        ])

    train_ds = ImageDataset(
        image_files=df_train.img_path, 
        labels=df_train[target], 
        transform=train_transforms)
    
    val_ds = ImageDataset(
        image_files=df_val.img_path, 
        labels=df_val[target], 
        transform=train_transforms)

    return train_ds,val_ds

