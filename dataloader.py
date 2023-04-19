import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import (
    Dataset,
)
from skimage import io
import os
import glob
from transforms import get_complete_transform,ContrastiveLearningViewGenerator
class CustomDataset(Dataset):

    def __init__(self, list_images, transform=None):
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.list_images[idx]
#         print(img_name)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image

output_shape = [224, 224]
kernel_size = [21,21] # 10% of the output_shape

# base SimCLR data augmentation
base_transforms = get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)

# The custom transform
custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms)

# complete dataset
trainn_ds = CustomDataset(
    list_images=glob.glob("/home/ccet/SIH/Data/ind-data/sirisplit/train/**/*.tif",recursive = True),
    transform=custom_transform
)

# train and valid dataset

train2_ds = ImageFolder(
    root="/home/ccet/SIH/Data/perc/siri-mulperc/10/train/",
    transform=custom_transform
)
valid2_ds = ImageFolder(
    root="/home/ccet/SIH/Data/perc/siri-mulperc/10/val/",
    transform=custom_transform
)
test2_ds = ImageFolder(
    root="/home/ccet/SIH/Data/perc/siri-mulperc/10/test/",
    transform=custom_transform
)

BATCH_SIZE = 256

# Building the data loader
train_dl = torch.utils.data.DataLoader(
    trainn_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)