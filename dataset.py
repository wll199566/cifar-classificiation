"""
Construct the dataset from raw data, which is at "./processed_data"
"""
import pickle

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class CIFAR10(Dataset):
    def __init__(self, data_path, dataset, data_aug=False):
        """
        data_path: folder storing train, valid and test set
        dataset: "train", "valid" or "test"
        data_aug: if use data augmentation, default: False
        """
        # initialize object variables
        self.data_aug = data_aug

        # read in the dataset
        with open(f"{data_path}/{dataset}_set.pkl", "rb") as fin:
            self.cifar_imgs = pickle.load(fin)

        # define transform
        # we keep the original image size 32*32, same as experiments in ResNet paper
        # Note that after ToTensor, the value of each pixel becomes [0,1]
        # then we can apply Normalize
        if data_aug:
            # Here the data augmentation is based on https://arxiv.org/pdf/1409.5185.pdf
            # Also, please take a look at PyTorch doc about how to solve dimension
            # problems due to tuples returned by TenCrop()
            self.transformations = transforms.Compose([
                transforms.Pad(padding=4),
                transforms.TenCrop(32, vertical_flip=False),  # return a tuple of 10 PIL images
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # convert the tuple into [B, C, H, W] 
                transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))(t) for t in tensors]))
                ])  # Note: valid and test should not use data aug
        else:
            self.transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])    
    
    def __getitem__(self, index):
        img = Image.fromarray(self.cifar_imgs[index][0]) # convert to PIL image
        label = self.cifar_imgs[index][1]
        
        # transform the image
        img = self.transformations(img)

        return img, label

    def __len__(self):  
        # notice that we cannot return the length of the list after data augmentation
        # otherwise, the sampler will sample from 1 to length after data augmentation
        # which will cause out of range error when getting items
        return len(self.cifar_imgs)  
    

if __name__ == "__main__":
    # initialize the dataset
    if_aug = True
    dataset = CIFAR10(data_path="./processed_data", dataset="valid", data_aug=if_aug)
    # build the data loader
    valid_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    # test it
    for idx, (img, label) in enumerate(valid_loader):
        if idx > 0:
            break
        # if using Tencrop, we need to 
        # (1) convert [batch, ncrops, c, h, w] into [batch*ncrops, c, h, w]
        # (2) interpolate the labels so that each augmented image can have a label
        if if_aug:
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label = torch.repeat_interleave(label, 10)

        print(img.shape, label)    
        #print(img, label)
        
