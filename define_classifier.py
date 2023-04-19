import torch
from config import DEVICE
from optimizer import LARS
import torch.nn as nn
from torch import optim
from define_simclr import simclr_model
from classifier import LinearEvaluation
from downstream_dataloader import nu_classes
simclr_model.load_state_dict(torch.load('simclr_pre_dataset9')) #load_tar
eval_model = LinearEvaluation(simclr_model, nu_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = LARS(eval_model.parameters(), lr=0.79, weight_decay=1e-6)