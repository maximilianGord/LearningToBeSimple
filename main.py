from data import getDataLoader
import torch
import numpy as np
from itertools import chain,combinations
from sklearn.metrics import confusion_matrix
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim




combined_generators_df = pd.read_csv('data_1.csv')
simple_indices = combined_generators_df.loc[combined_generators_df.y==True].index
size_simple = len(simple_indices)
size_df = len(combined_generators_df.index)
size_not_simple = size_df-size_simple
print("Raw number of simple groups : "+str(size_simple))
#make sure the whole Dataset is balanced
residual = size_not_simple%4
# We want to split the data into 1/5 simple and 4/5 nonsimple or in other words we split 1:4. Before that we have to make the number divisible 
if residual!=0:
    size_not_simple -= residual
    size_df         -= residual
    combined_generators_df = combined_generators_df.loc[pd.Index(np.random.choice(
        combined_generators_df.loc[combined_generators_df.y==False].index,size=size_not_simple,replace=False
    )).append(combined_generators_df.loc[combined_generators_df.y==True].index).sort_values()] 
# If to many simple groups : remove some simple groups and keep the non simple
if size_simple>size_not_simple/4:
    size_df -=(size_simple-int(size_not_simple/4))
    size_simple = int(size_not_simple/4)
    combined_generators_df = combined_generators_df.loc[pd.Index(np.random.choice(
        combined_generators_df.loc[combined_generators_df.y==True].index,size=size_simple,replace=False
    )).append(combined_generators_df.loc[combined_generators_df.y==False].index).sort_values()] 
# If to many non simple groups : remove some non simple groups and keep the simple
else:
    size_df -= (size_not_simple-size_simple*4)
    size_not_simple = size_simple*4 
    combined_generators_df = combined_generators_df.loc[pd.Index(np.random.choice(
        combined_generators_df.loc[combined_generators_df.y==False].index,size=size_not_simple,replace=False
    )).append(combined_generators_df.loc[combined_generators_df.y==True].index).sort_values()]
print("Processed number of simple groups : "+str(size_simple)+" and processed size of data : "+str(size_df))
train_data = getDataLoader(combined_generators_df)
order = 5
# Define Model and Train Function, use optimised number of hidden layers from Paper
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(2*order**2,256)
        self.output_layer = nn.Linear(256,2)
    def forward(self, x):
        x = nn.functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return nn.functional.softmax(x,dim=1)
# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.to(torch.float32))
        loss = nn.functional.mse_loss(output.argmax(dim=1, keepdim=True).flatten().float(), target)
        loss.backward()
        optimizer.step()
k_folds = 5
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(skf.split(train_data[0],train_data[1])):
    train_loader = DataLoader(
        dataset=list(zip(train_data[0],train_data[1])),
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    test_loader = DataLoader(
        dataset=list(zip(train_data[0],train_data[1])),
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    for epoch in range(1, 30):
        train(model, device, train_loader, optimizer, epoch)
    model.eval()
    test_loss = 0
    correct = 0
    all_preds=[]
    all_targets=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.to(torch.float32))
            test_loss += nn.functional.mse_loss(output.argmax(dim=1, keepdim=True).flatten().float(), target,reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.view_as(target).tolist())
            all_targets.extend(target.tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print(pred.eq(target.view_as(pred)).sum().item()/len(target))
    #all_preds=np.array(all_preds)
    #all_targets=np.array(all_targets)
    cm=confusion_matrix(all_targets,all_preds)
    TP = cm[1, 1]  # True Positives
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives

    # Print individual counts
    print("------------- Fold : {fold}-------------")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    test_loss /= len(test_idx)
    accuracy = 100.0 * correct / len(test_idx)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_idx)} ({accuracy:.2f}%)\n")










