"""
Construct the dataset from raw data, which is at "./processed_data"
"""
import pickle

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class CIFAR10(Dataset):
    def __init__(self, data_path, dataset):
        """
        data_path: folder storing train, valid and test set
        dataset: "train", "valid" or "test"
        """
        # read in the dataset
        with open(f"{data_path}/{dataset}_set.pkl", "rb") as fin:
            self.cifar_imgs = pickle.load(fin)

        # define transform
        # we keep the original image size 32*32, same as experiments in ResNet paper
        # Note that after ToTensor, the value of each pixel becomes [0,1]
        # then we can apply Normalize
        self.transformations = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    
    def __getitem__(self, index):
        img = Image.fromarray(self.cifar_imgs[index][0]) # convert to PIL image
        label = self.cifar_imgs[index][1]
        
        # transform the image
        img = self.transformations(img)

        return img, label

    def __len__(self):
        return len(self.cifar_imgs)    
    

if __name__ == "__main__":
    # initialize the dataset
    dataset = CIFAR10(data_path="./processed_data", dataset="valid")
    # build the data loader
    valid_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    # test it
    for idx, (img, label) in enumerate(valid_loader):
        if idx > 0:
            break
        print(img.shape, label)    

        
