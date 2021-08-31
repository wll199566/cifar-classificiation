"""
Tune Hyperparameters with Tensorboard
------------------------------------------------------------------------
Codes are modified from https://github.com/ajinkya98/TensorBoard_PyTorch
------------------------------------------------------------------------
Search strategy (CS231N Notes and slides):
1. random search
2. from coarser to finer (first use greedy alg to get the sense of ranges)
3. learning rate is the most important one
4. for coarser, use shorter epochs; for finer, use longer epochs
"""
import time
import argparse
import logging
import pickle
import random
from datetime import datetime
from itertools import product

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from dataset import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CIFARResNet, MLP
from model import init_weights

from utils import new_dir


### reproducibility ###
# https://pytorch.org/docs/stable/notes/randomness.html
seed = 1234
#random.seed(seed) # we want different sampled hyperparameters each time
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # more efficient when multi-gpu
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

### configurations ###
# non-searchable (fixed)
data_path = "./processed_data"
model_name = "resnet"
n = 3
num_epochs = 5  # can be adjust, for coarse search, it can be shorter, for finer search, it can be longer
momentum = 0.9

# searchable (to-be-searched) hyperparameters
# here, we need to manully input the searchable hyperparameters
# so that we can run them in parallel in order to save a lot of time
def arg_parser():
    """
    Parse arguments 
    """
    parser = argparse.ArgumentParser(description='hyperparameter tuning')
    parser.add_argument("--daug", type=bool, default=False, help="data augmentation")
    # searchable hyperparameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument

    args = parser.parse_args()
    
    return args

# get the searchable hyperparameters from the arguments to get the intuition of the ranges
# we use the greedy algorithm to find ranges according to the following order
args = arg_parser()
lr = args.lr
batch_size = args.batch_size
weight_decay = args.weight_decay

# Or, we can directly sample here without arguments for the finer search using the ranges obtained above
#lr = 10 ** random.uniform(-4, 0)
#batch_size = random.choice([64, 128, 256])
#weight_decay = 10 ** random.uniform(-4, 0)    

### iterate all combinations of all the hyparameter values ###
#param_values_list = [v for v in search_param_values.values()]
#for run_id, (lr, batch_size, weight_decay) in enumerate(product(*param_values_list)):
    
### explicit the comment name ###
if args.daug:
    comment_name = f"_lr_{lr}_batch_{batch_size}_wd_{weight_decay}_daug"
else:
    comment_name = f"_lr_{lr}_batch_{batch_size}_wd_{weight_decay}"
# initialize tensorboard writer
writer = SummaryWriter(comment=comment_name)

### prepare for training and validation ###
# initialize components
### initialize essentials for train and valid ###
# dataloader
train_loader = DataLoader(CIFAR10(data_path=data_path, dataset="train", data_aug=args.daug), batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(CIFAR10(data_path=data_path, dataset="valid"), batch_size=batch_size, shuffle=False, pin_memory=True)
# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model
if model_name == "mlp":
    model=MLP()
elif model_name == "resnet":    
    model = CIFARResNet(block_num=n)
else:
    raise NameError(f"{model_name} is not implemented!")
# initialize weight and bias
model = init_weights(model)
# model parallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
# criterion
criterion = nn.CrossEntropyLoss(reduction="sum")
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# used to record the best validation accuracy
best_val_acc = 0  

### each epoch ###
for epoch in range(1, num_epochs+1):
    ### train ###
    # loss and correct is not averaged due to the probable different batch size of
    # the last batch
    model.train()
    total_count = 0  # total training images
    train_total_loss = 0  # total loss of every epoch (not average)
    train_total_correct = 0  # total correct predictions every epoch
    train_start_time = time.time()  # record train start time
    for batch, (imgs, labels) in enumerate(train_loader, 1):
        # clean up the buffered gradients
        model.zero_grad()
        # if data augmentation, 
        if args.daug:
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label = torch.repeat_interleave(label, 10)
        # put data into device
        imgs, labels = imgs.to(device), labels.to(device)
        # forward pass
        preds = model(imgs)
        train_loss = criterion(preds, labels)
        # backward pass
        train_loss.backward()
        optimizer.step()
        # update loss and correct counts
        train_count += len(imgs)
        train_total_loss += train_loss.item()
        train_correct = torch.sum(torch.argmax(preds, dim=1) == labels).item()
        train_total_correct += train_correct
    # write training loss and accuracy into Tensorboard
    train_avg_loss = train_total_loss/train_count
    train_acc = train_total_correct/train_count
    writer.add_scalar("train_loss", train_avg_loss, epoch)
    writer.add_scalar("train_acc", train_acc, epoch)
    
    ### Validation ###
    model.eval()
    valid_total_loss = 0
    valid_total_correct = 0
    valid_start_time = time.time()
    with torch.no_grad():
        for batch, (imgs, labels) in enumerate(valid_loader):
            # put data into device
            imgs, labels = imgs.to(device), labels.to(device)
            # forward pass
            preds = model(imgs)
            valid_loss = criterion(preds, labels)
            # update loss and correct counts
            valid_total_loss += valid_loss.item()
            valid_total_correct += torch.sum(torch.argmax(preds, dim=1) == labels).item() 
    # write into tensorboard 
    valid_avg_loss = valid_total_loss/len(valid_loader.dataset)
    valid_acc = valid_total_correct/len(valid_loader.dataset)
    writer.add_scalar("valid_loss", valid_avg_loss, epoch)
    writer.add_scalar("valid_acc", valid_acc, epoch)                        

    # update learning rate if necessary
    # we use valid acc based on VGG paper (https://arxiv.org/pdf/1409.1556.pdf)
    if scheduler != None:
        scheduler.step(valid_acc)

    # default: save the model with the best parameters
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc

### add the parameter values and valid result into the tensorboard ###
writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay},
        {"best_valid_acc": best_val_acc}
    )

# close the tensorboard writer
writer.close()      