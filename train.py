"""
Train and Validation
"""
import time
import argparse
import logging
import pickle
from datetime import datetime

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

def arg_parser():
    """
    Parse arguments 
    """
    parser = argparse.ArgumentParser(description='configurations')
    ### systems ###
    parser.add_argument("--data_path", type=str, default="./processed_data")
    parser.add_argument("--seed", type=int, default=1234)
    ### hyparameters ###
    # model config 
    parser.add_argument("--model", type=str)
    parser.add_argument("--n", type=int, default=3, 
    help="# of layers each feature map has")
    # hyperparameters
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0)
    ### others ###
    parser.add_argument("--save_freq", type=int, default=0, 
    help="how frequent to save model and optimizer status (0 means no saving)")
    parser.add_argument("--resume_epoch", type=int, default=0, 
    help="from which epoch to resume the training from the checkpoint (0 means train from scratch)")
    parser.add_argument("--verbose", action="store_true", 
    help="to print training loss for each batch")
    args = parser.parse_args()
    
    return args

def train_valid(args, data_loader:tuple, device, model, criterion, optimizer, scheduler=None):
    """
    Train and validation
    data_loader: tuple of (train_loader, valid_loader)
    """ 
    ### Prepare for recording the experiment results ###
    # create a new folder for the current model
    new_dir("./records")
    record_hpp_str = f"{args.model}_n_{args.n}_lr_{args.lr}_wd_{args.weight_decay}_mmt_{args.momentum}" # for create path
    model_root_path = f"./records/{record_hpp_str}"
    new_dir(model_root_path)
    # create a directory to store model and optimizer status
    if args.save_freq > 0:
        new_dir(f"{model_root_path}/models")
    # initialize tensorboard writer
    writer = SummaryWriter(f"{model_root_path}/tensorboard")    
    # config the logging
    logging.basicConfig(filename=f"{model_root_path}/train_valid_log.txt", format='%(message)s', filemode="w", level=logging.INFO)
    logging.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logging.info("\n")
    logging.info("Experiment settings")
    logging.info("-"*100)
    logging.info(args)
    logging.info(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"{torch.cuda.device_count()} GPUs are used!")
    logging.info("\n")    
    # dictionary to record the train and valid history
    history = {"train_loss": [], "valid_loss":[], "train_acc":[], "valid_acc":[]}
    # used to record the best validation accuracy
    best_val_acc = 0  
    

    ### each epoch ###
    for epoch in range(1, args.epochs+1):

        logging.info(f"Epoch {epoch}")
        logging.info("-"*100)
        
        ### train ###
        # loss and correct is not averaged due to the probable different batch size of
        # the last batch
        model.train()
        train_total_loss = 0  # total loss of every epoch (not average)
        train_total_correct = 0  # total correct predictions every epoch
        train_start_time = time.time()  # record train start time
        for batch, (imgs, labels) in enumerate(data_loader[0]):
            # clean up the buffered gradients
            model.zero_grad()
            # put data into device
            imgs, labels = imgs.to(device), labels.to(device)
            # forward pass
            preds = model(imgs)
            train_loss = criterion(preds, labels)
            # backward pass
            train_loss.backward()
            optimizer.step()
            # update loss and correct counts
            train_total_loss += train_loss.item()
            train_total_correct += torch.sum(torch.argmax(preds, dim=1) == labels).item()
            if args.verbose:
                logging.info(f"epoch {epoch} | batch {batch} | train_loss: {train_loss.item()}")   
        # logging loss and accuracy
        train_avg_loss = train_total_loss/len(data_loader[0].dataset)
        train_acc = train_total_correct/len(data_loader[0].dataset)
        train_avg_time = (time.time()-train_start_time)/len(data_loader[0].dataset)
        logging.info(f"train avg loss: {train_avg_loss:.3f} \
                     | train accuracy: {train_acc:.3f} \
                     | train avg time: {train_avg_time:.5f}s")
        history["train_loss"].append(train_avg_loss)
        history["train_acc"].append(train_acc)
        
        ### Validation ###
        model.eval()
        valid_total_loss = 0
        valid_total_correct = 0
        valid_start_time = time.time()
        with torch.no_grad():
            for batch, (imgs, labels) in enumerate(data_loader[1]):
                # put data into device
                imgs, labels = imgs.to(device), labels.to(device)
                # forward pass
                preds = model(imgs)
                valid_loss = criterion(preds, labels)
                # update loss and correct counts
                valid_total_loss += valid_loss.item()
                valid_total_correct += torch.sum(torch.argmax(preds, dim=1) == labels).item() 
        # logging 
        valid_avg_loss = valid_total_loss/len(data_loader[1].dataset)
        valid_acc = valid_total_correct/len(data_loader[1].dataset)
        valid_avg_time = (time.time()-valid_start_time)/len(data_loader[1].dataset)
        logging.info(f"valid avg loss: {valid_avg_loss:.3f} \
                     | valid accuracy: {valid_acc:.3f} \
                     | valid avg time: {valid_avg_time:.5f}s")
        history["valid_loss"].append(valid_total_loss/len(data_loader[1].dataset))
        history["valid_acc"].append(valid_total_correct/len(data_loader[1].dataset))
        logging.info("\n")  

        # write into tensorboard
        # train and valid loss into one plot, train and valid eval metrics into another plot
        writer.add_scalars("loss", {"train_loss": train_avg_loss,
                                    "valid_loss": valid_avg_loss}, epoch)
        writer.add_scalars("acc", {"train_acc": train_acc,
                                   "valid_acc": valid_acc}, epoch)                          

        # update learning rate if necessary
        # we use valid acc based on VGG paper (https://arxiv.org/pdf/1409.1556.pdf)
        if scheduler != None:
            scheduler.step(valid_acc)

        # default: save the model with the best parameters
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save({
                "epoch": epoch,
                "best_acc": best_val_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, f"{model_root_path}/best_model.pth")
        
        # save the model and optimizer states
        if args.save_freq >  0:
            if epoch / args.save_freq == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, f"{model_root_path}/models/epoch_{epoch}.pth")

    # write history into the file
    with open(f"{model_root_path}/history.pkl", "wb") as fout:
        pickle.dump(history, fout)  

    # close the tensorboard writer
    writer.close()      

        
if __name__ == "__main__":

    ### parse arguments ###
    args = arg_parser()
    
    ### reproducibility ###
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # more efficient when multi-gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ### initialize essentials for train and valid ###
    # dataloader
    train_loader = DataLoader(CIFAR10(data_path=args.data_path, dataset="train"), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(CIFAR10(data_path=args.data_path, dataset="valid"), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model
    if args.model == "mlp":
        model=MLP()
    elif args.model == "resnet":    
        model = CIFARResNet(block_num=args.n)
    else:
        raise NameError(f"{args.model} is not implemented!")
    # initialize weight and bias
    model = init_weights(model)
    # model parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    # criterion
    criterion = nn.CrossEntropyLoss(reduction="sum")
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    ### train ###
    train_valid(args, (train_loader, valid_loader), device, model, criterion, optimizer, scheduler)
    
