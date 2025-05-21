import os
import pandas as pd
import numpy as np
from data_loader import construct_dataframe
from visualization import visualize, visualize_result
from SARdataset import SARdataset, transform, test_transform
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import nn
from model_and_training import get_model, iou_score, train_model
from test import get_pred

# set data directory
data_dir = "/content/ETCI_2021_Competition_Dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# get dataframes containing vv, vh, floodmask, waterbodymask
train_df = construct_dataframe(train_dir, include_flood_label=True)
val_df = construct_dataframe(val_dir, include_flood_label=False)

# Visualize some samples
visualize(train_df.iloc[0])
visualize(val_df.iloc[0])

# all regions in the training set
regions = ['nebraska', 'northal', 'bangladesh']

# randomly choose one for the development set and leave the rest for training
development_region = np.random.choice(regions, 1)[0]
regions.remove(development_region)
train_regions = regions

# filter the dataframe to only get images from specified regions
sub_train_df = train_df[train_df['region'] != development_region]
development_df = train_df[train_df['region'] == development_region]

# check that new dataframes only have the image paths from the correct regions
print('Sub-training set regions: {}'.format(np.unique(sub_train_df['region'].tolist())))
print('Development set region: {}'.format(np.unique(development_df['region'].tolist())))

# construct dataset
train_dataset  = SARdataset(sub_train_df, split='train', transform=transform)
dev_dataset = SARdataset(development_df, split='dev')

# construct loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# define device gpu/cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## define model 
model = get_model(num_classes=2)
model.to(device)

## define optimizer/scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7,gamma=0.1)

## define loss function

criterion = nn.CrossEntropyLoss()

# enter training loop
num_epochs = 5
model = train_model(device, model, train_loader, dev_loader, num_epochs, iou_score, optimizer, scheduler, criterion num_epochs, iou_score, optimizer, scheduler, criterion):

# save model
torch.save(model.state_dict(), 'model.pt')

# load model
model = get_model()
model.load_state_dict(torch.load('model.pt'))
model.to(device)
model.eval()

# test model

test_dataset = SARdataset(val_df, split='test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

final_predictions = []
fianl_predictions = get_pred(device, model, test_loader, final_predictions)


# some test visualizations
index = -1910
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 570
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 68
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = -1919
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = -1
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 498
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 75
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 395
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 4097
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))

index = 8999
visualize_result(val_df.iloc[index], final_predictions[index], figsize=(17,10))
