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
    imgs, labels = next(iter(train_loader))
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




if __name__ == "__main__":
    ### Configuration ###
    seed = 1234
    data_path = "./processed_data"
    batch_size = 256  # used for check loss and overfit small data
    train_model = "resnet"
    N_block = 3
    class_weight = None  # used to define the loss function and compute the loss of a random classifier
    n_class = 5  # how many classes in the labels

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
    
    # check 1: correct loss
    check_init_loss(model, criterion, train_loader, device, n_class)
        

