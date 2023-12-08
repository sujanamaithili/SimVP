import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

class SegmentationDataSet(Dataset):

    def __init__(self, args,transform=None):

        self.stored_images_path=args.res_dir+'/Debug/results/Debug/sv/last_frames.npy'

        print("last frames stored path::",self.stored_images_path)

        self.last_frames = np.load(self.stored_images_path) #(2000,1,3,160,240)
        
        print("last frames shape:", self.last_frames.shape)

    def __len__(self):
        return len(self.last_frames)

    def __getitem__(self, index):
        return self.last_frames[index]  # we want to return (3,160,240) this dimension

class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(self, out_channels=49, features=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = encoding_block(3, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block(features[2], features[3])
        self.conv5 = encoding_block(features[3] * 2, features[3])
        self.conv6 = encoding_block(features[3], features[2])
        self.conv7 = encoding_block(features[2], features[1])
        self.conv8 = encoding_block(features[1], features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3], features[3] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x


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
print("Jaccard Index:", mean_jaccard_index)
