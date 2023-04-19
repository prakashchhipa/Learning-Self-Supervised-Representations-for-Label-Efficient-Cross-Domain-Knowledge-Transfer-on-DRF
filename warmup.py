from define_classifier import eval_model,optimizer,scheduler,criterion
from config import DEVICE
from downstream_dataloader import train_dl
import time
cntr=0
for child in eval_model.children():
    cntr+=1
    print(child)

eval_model.children()

ct = 0
for child in eval_model.children():
    ct += 1
    if ct < 2:
        for param in child.parameters():
            param.requires_grad = False

from tqdm import tqdm
EPOCHS = 10
for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    for i, element in enumerate(tqdm(train_dl)):
        image, label = element
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        pred = eval_model(image)
#         print(image.shape)
#         print(pred.shape)
#         print(label.shape)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    if(epoch % 10 == 0) :                        
        print(f'EPOCH: {epoch+1} BATCH: {i+1} LOSS: {(running_loss/100):.4f} ')
    running_loss = 0.0
#     print(f'Time taken: {((time.time()-t0)/60):.3f} mins')

ct = 0
for child in eval_model.children():
    ct += 1
    if ct <= 2:
        for param in child.parameters():
            param.requires_grad = True