import torch
import numpy as np
import torchmetrics
from os import listdir
from os.path import isfile, join

y_pred_masks = np.load('/scratch/sc10648/Unet/vanillaSimVP/Pipeline/results/numpy_y_pred_masks.npy')

# Convert numpy array to torch tensors
y_pred_masks_tensor = torch.tensor(y_pred_masks, dtype=torch.int32)

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

val_directory = '/scratch/sc10648/DL-Competition1/dataset/val'

# List to store Jaccard indices
jaccard_indices = []

# Loop through each validation directory
for i in range(1000, 2000):
    folder_path = join(val_directory, f'video_{i}')
    mask_path = join(folder_path, 'mask.npy')

    # Load the 22nd mask from the validation data
    val_mask = np.load(mask_path)[21]  # 22nd mask is at index 21
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.int32)

    # Compute Jaccard index
    jaccard_index_value = jaccard(y_pred_masks_tensor[i - 1000].unsqueeze(0), val_mask_tensor.unsqueeze(0))
    jaccard_indices.append(jaccard_index_value.item())

# Calculate mean Jaccard index
mean_jaccard_index = np.mean(jaccard_indices)
Print("Jaccard Index:", mean_jaccard_index)

