import torch
import os
from dataloader import train2_ds,test2_ds,valid2_ds
nu_classes = 5 #number of classes in the target dataset

BATCH_SIZE = 256

# train_size = int(0.8 * len(ds))
# valid_size = len(ds) - train_size
# train_ds, valid_ds = torch.utils.data.random_split(ds, [train_size, valid_size])

# print(len(train_ds))
# print(len(valid_ds))

# Building the data loader
train_dl = torch.utils.data.DataLoader(
    train2_ds,
    batch_size=128,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

valid_dl = torch.utils.data.DataLoader(
    valid2_ds,
    batch_size=32,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)
test_dl = torch.utils.data.DataLoader(
    test2_ds,
    batch_size=16,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)