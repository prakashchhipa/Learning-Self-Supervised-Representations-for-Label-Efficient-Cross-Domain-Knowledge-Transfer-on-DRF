from tqdm import tqdm
from define_simclr import simclr_model
from contrastive_loss import cont_loss
from config import DEVICE
from define_simclr import scheduler,optimizer,criterion
import torch
import time
from downstream_dataloader import train_dl
EPOCHS = 101
checkpoint = 1
for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    for i, views in enumerate(tqdm(train_dl)):
        projections = simclr_model([view.to(DEVICE) for view in views])
        logits, labels = cont_loss(projections, temp=0.5)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
        # print statistics
    if(epoch % 10 == 0) :                        
        print(f'EPOCH: {epoch+1} BATCH: {i+1} LOSS: {(running_loss/100):.4f} ')
        
        # Checkpoint 
        torch.save(simclr_model.state_dict(), f'simclr_resnet50_pre_two_stage_{checkpoint}') 
        checkpoint += 1 
    running_loss = 0.0
    print(f'Time taken: {((time.time()-t0)/60):.3f} mins')