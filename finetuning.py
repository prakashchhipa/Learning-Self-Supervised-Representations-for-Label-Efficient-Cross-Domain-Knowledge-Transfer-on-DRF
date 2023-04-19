from config import DEVICE
import torch
from define_classifier import criterion,eval_model,optimizer,scheduler
from downstream_dataloader import train_dl,valid_dl
train_loss = list()
train_acc= list()
val_loss= list()
val_acc= list()
from tqdm import tqdm
epochs = 100
for epoch in range(epochs):
    accuracies = list()
    class_losses = list()
    eval_model.train()
    for class_batch in tqdm(train_dl):
        x, y = class_batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
       
        logit = eval_model(x)
        classification_loss = criterion(logit, y)
        class_losses.append(classification_loss.item())

        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()
        accuracies.append(y.eq(logit.detach().argmax(dim =1)).float().mean())
    scheduler.step()  
    if (epoch+1)%5==0:
        torch.save(eval_model.state_dict(), f'cl1_simclreval_newdata_{epoch+1}')
        print(f"saved checkpoint for epoch {epoch + 1}")

    train_loss.append(class_losses)
    train_acc.append(accuracies)
    print(f'Epoch {epoch + 1}')
    print(f'classification training loss: {torch.tensor(class_losses).mean():.5f}')
    print(f'classification training accuracy: {torch.tensor(accuracies).mean():.5f}', 
          end ='\n\n')
    

    losses = list()
    accuracies = list()
    eval_model.eval()
    for batch in tqdm(valid_dl):
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            logit =eval_model(x)

        loss = criterion(logit, y)

        losses.append(loss.item())
        accuracies.append(y.eq(logit.detach().argmax(dim =1)).float().mean())
    
    val_loss.append(losses)
    val_acc.append(accuracies)
    print(f'Epoch {epoch + 1}')
    print(f'classification validation loss: {torch.tensor(losses).mean():.5f}')
    print(f'classification validation accuracy: {torch.tensor(accuracies).mean():.5f}', 
          end ='\n\n')
