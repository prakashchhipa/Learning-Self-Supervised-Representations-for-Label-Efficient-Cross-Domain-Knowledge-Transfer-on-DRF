from torch import optim
from config import DEVICE
import torch.nn as nn
import torch
from optimizer import LARS
from simclr_model import SimCLR
simclr_model = SimCLR().to(DEVICE)       # network model
criterion = nn.CrossEntropyLoss().to(DEVICE)        # loss
# optimizer = torch.optim.Adam(simclr_model.parameters(), lr=0.79, weight_decay=1e-6)     # optimizer
# optimizer = LARS(optim.SGD(clr_model.parameters(), lr=0.59, weight_decay=1e-6))
optimizer = LARS(simclr_model.parameters(), lr= 0.79, weight_decay=1e-6)