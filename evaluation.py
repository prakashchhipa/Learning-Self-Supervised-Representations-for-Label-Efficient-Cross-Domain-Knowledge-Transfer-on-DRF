from define_classifier import eval_model
import torch
from config import DEVICE
from tqdm import tqdm
import matplotlib.pyplot as plt
from downstream_dataloader import test_dl
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame
import seaborn as sn
import pandas as pd
eval_model.load_state_dict(torch.load('cl1_simclreval_newdata_385')) #load_tar
correct = 0
total = 0
preds = []
labels = []
with torch.no_grad():
    for i, element in enumerate(tqdm(test_dl)):
        image, label = element
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        outputs = eval_model(image)
        _, predicted = torch.max(outputs.data, 1)
        preds += predicted.cpu().numpy().tolist()
        labels += label.cpu().numpy().tolist()
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Accuracy: {100 * correct / total} %')
confm = confusion_matrix(labels, preds)
#confusion_matrix_df = pd.DataFrame(confusion_matrix(labels, preds))
target_names = ['0', '1', '2', '3', '4']
print(classification_report(labels, preds, target_names=target_names))
confusion_matrix_df = pd.DataFrame(confusion_matrix(labels, preds))
print(cohen_kappa_score(labels, preds, weights='quadratic'))
plt.figure(figsize = (10,10),dpi=120)
sn.heatmap(confusion_matrix_df, annot=True, cmap="Blues_r")
# probs from log preds
#labels.reshape(-1,1)
probs = np.exp(labels[:])
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(labels, preds)

# Compute ROC area
roc_auc = auc(fpr, tpr)
print('ROC area is {0}'.format(roc_auc))
#print(auc(fpr,tpr))
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.8f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
