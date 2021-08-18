"""
This script is for the sanity check for the model based on Andrej Karpathy's blog:
A Recipe to Train Neural Networks (http://karpathy.github.io/2019/04/25/recipe/),
and his CS231N notes: https://cs231n.github.io/neural-networks-3/
Note at this stage, we need to turn off all the fancy techniques, e.g. data augmentation,
regularization, etc.
"""

import torch
import torch.nn as nn
from torch import optim
from dataset import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from model import CIFARResNet, MLP
from model import init_weights

# check 1: correct loss 
# e.g. for classification with softmax and CE loss, the loss cannot be 
# larger than -log(1/n_classes)*class_weight 
def check_init_loss(model, criterion, loader, device, num_class):
    # here, we try to feed the model with only one batch of data 
    model.train()
    model.zero_grad()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    # compute the init loss
    preds = model(imgs)
    train_loss = criterion(preds, labels).item()
    # compute the loss of the random classifier
    preds_rand = 1/num_class * torch.ones_like(preds, device=device)  
    random_loss = criterion(preds_rand, labels).item()
    # print the information
    print(f"Init avg train loss for one batch ({len(imgs)}):") 
    print(f"random classifier: {random_loss/len(imgs):.6f} | model: {train_loss/len(imgs):.6f}")
    # a correct model and initialization should make these two close to each other.

# check 2: 
# (1) if the training loss is increasing with the full training data with increasing regularization
# ==> if yes, then the model is underfitting
# But note if the loss is becoming NaN, then the learning rate might be too large
# (2) if the training loss can decrease with the full training data with increasing model capacity
# ==> if yes, then our model is not underfitting
# we can check for only a few epoches for a quick check
def check_loss_dec(model, criterion, loader, device, optimizer):
    model.train()
    for epoch in range(1, 11):
        print(f"Epoch {epoch}")
        epoch_loss = 0
        for batch, (imgs, labels) in enumerate(loader, 1):
            model.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            # forward pass
            preds = model(imgs)
            batch_loss = criterion(preds, labels)
            # backward pass
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            if batch % 10 == 0:
                print(f"epoch{epoch} | batch{batch} | batch loss: {batch_loss/len(imgs)}")
        
        print(f"epoch loss: {epoch_loss/len(loader.dataset)}")    
        print("-"*100 + "\n")       
        

# check 3: if the model has enough capacity to overfit 2 samples (zero loss)
def overfit_small_data(num_epoch, model, criterion, loader, device, optimizer):
    model.train()
    # get the training samples (first 2 training samples)
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:2,...].to(device), labels[:2,...].to(device)
    # begin our training
    for epoch in range(1, num_epoch+1):
        model.zero_grad()
        # forward pass
        preds = model(imgs)
        loss = criterion(preds, labels)
        # backward pass
        loss.backward()
        optimizer.step()
        # compute train loss and train acc
        train_loss = loss.item()
        train_acc = torch.sum(torch.argmax(preds, dim=1) == labels).item()
        print(f"epoch: {epoch} | train_loss: {train_loss/len(imgs)} | train_acc: {train_acc/len(imgs)}")
        

if __name__ == "__main__":
    ### Configuration ###
    seed = 1234
    data_path = "./processed_data"
    batch_size = 256  # used for check loss and overfit small data
    train_model = "resnet"
    N_block = 3
    class_weight = None  # used to define the loss function and compute the loss of a random classifier
    n_class = 5  # how many classes in the labels
    n_epoch = 30  # used for decreasing loss and ovefitting small data
    learn_rate = 0.001

    ### reproducibility ###
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # more efficient when multi-gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ### initialize essentials for train and valid ###
    # dataloader
    train_loader = DataLoader(CIFAR10(data_path=data_path, dataset="train"), batch_size=batch_size, shuffle=True, pin_memory=True)
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model
    if train_model == "mlp":
        model=MLP()
    elif train_model == "resnet":    
        model = CIFARResNet(block_num=N_block)
    else:
        raise NameError(f"{train_model} is not implemented!")
    # initialize weight and bias
    model = init_weights(model)
    # model parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    # criterion
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="sum")
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)
    
    ### Sanity Check ###
    ### check 1: correct loss ###
    #check_init_loss(model, criterion, train_loader, device, n_class)

    ### check 2: check if loss can decrease ###
    check_loss_dec(model, criterion, train_loader, device, optimizer)

    ### check 3: overfit_small_data ###
    #overfit_small_data(n_epoch, model, criterion, train_loader, device, optimizer)
        

