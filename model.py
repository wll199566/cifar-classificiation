"""
This model is from the ResNet paper on CIFAR-10 dataset
Here, we implement all three models
----------------------------------------------------------
Model architecture:
Inputs: 32*32 images
Output: 10 classes
Architecture:
3*3 conv layer + BatchNorma + 32 channels
for # of filters of each conv {16, 32, 64}:
    2n * (3*3 conv + BatchNorm + option A identity mapping)
Global average layer
1000 linear layer

Downsample: 3*3 conv layer with strides 2

n can be {3, 5, 7, 9} in the paper
"""
import torch
import torch.nn as nn
from dataset import CIFAR10
from torch.utils.data import DataLoader

class BasicBlock(nn.Module):
    """
    Basic Block is 2 3*3 conv layers with a shortcut connection
    """
    def __init__(self, in_channels:int) -> None:
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=in_channels)
            
    def forward(self, x):
        # compute the residual
        x_residual = self.relu(self.batchnorm1(self.conv1(x)))
        x_residual = self.relu(self.batchnorm2(self.conv2(x)))
        assert x.shape == x_residual.shape, "x shape is different from x_residual shape"
        # combine the residual with the input
        return x + x_residual
        

class DownSampleBlock(nn.Module):
    """
    Downsample block, 3*3 conv layer with stride 2 and padded shortcut connection
    """
    def __init__(self, in_channels:int) -> None:
        super(DownSampleBlock, self).__init__()
        self.pad = nn.ZeroPad2d((0,1,0,1))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=3, stride=2, padding=0)  
        self.batchnorm1 = nn.BatchNorm2d(num_features=2*in_channels)
        self.conv2 = nn.Conv2d(in_channels=2*in_channels, out_channels=2*in_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=2*in_channels)
        # Note that all downsamples for option A or B is implemented by 1*1 conv layer
        # Also, there is no ReLU in shortcut
        self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=1, stride=2, padding=0)
        self.batchnorm_shortcut = nn.BatchNorm2d(num_features=2*in_channels)
        
    def forward(self, x):
        # compute the residual
        x_residual = self.pad(x) # pad height and width by 1 
        x_residual = self.relu(self.batchnorm1(self.conv1(x_residual)))
        x_residual = self.relu(self.batchnorm2(self.conv2(x_residual)))
        # compute the shortcut
        x_shortcut = self.batchnorm_shortcut(self.conv_shortcut(x))
        assert x_shortcut.shape == x_residual.shape, "x shape is different from x_residual shape"
        # combine the residual with the input
        return x_shortcut + x_residual


class CIFARResNet(nn.Module):
    def __init__(self, block_num):
        """
        block_num: # of basic/downsample block each feature map has
        """
        super(CIFARResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )
        layers = [] # add layers from BasicBlock and DownSampleBlock
        # feature maps: 16, map size: 32*32
        for i in range(block_num):
            layers.append(BasicBlock(16))
        # feature maps: 32, map size: 16*16
        layers.append(DownSampleBlock(16))
        for i in range(block_num-1):
            layers.append(BasicBlock(32))
        # feature maps: 64, map size: 8*8
        layers.append(DownSampleBlock(32))
        for i in range(block_num-1):
            layers.append(BasicBlock(64))
        self.block = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)
        # no need to use Softmax, since we will use
        # CrossEntropyLoss which combines LogSoftmax and NLLLoss

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(32*32*3, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x

def init_weights(model):
    """
    Initialize weights and bias of all layers. Implement it as a 
    separate methods so it can be applied to every model defined 
    in this file
    """        
    # the implementation is copied from 
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)  # suggested by https://cs231n.github.io/neural-networks-2/#init
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    return model                


if __name__ == "__main__":
    
    ### test for the subblock ###
    # create a input
    # input_feat = torch.ones(10, 16, 32, 32)
    # # build a model
    # model_basic = BasicBlock(16)
    # model_down = DownSampleBlock(16)
    # output_basic = model_basic(input_feat)
    # output_down = model_down(input_feat)
    # print(output_basic.shape)
    # print(output_down.shape)

    # initialize the dataset
    dataset = CIFAR10(data_path="./processed_data", dataset="valid")
    # build the data loader
    valid_loader = DataLoader(dataset, batch_size=5, shuffle=False)
    # build a model
    #model = MLP()
    model = CIFARResNet(3)
    # test for the initialization
    #print(model)
    # take a look at the initialized bias
    model=init_weights(model)
    print(model.block[0].batchnorm1.weight)
    print(model.block[0].batchnorm1.bias)
    print(model.block[0].conv1.weight)
    print(model.block[0].conv1.bias)


    # # put it into device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model.to(device)
    # # loss function
    # criterion = nn.CrossEntropyLoss()
    
    # # test it
    # for idx, (img, label) in enumerate(valid_loader):
    #     img, label = img.to(device), label.to(device)
    #     if idx > 0:
    #         break
    #     output = model(img)
    #     # compute the loss
    #     loss = criterion(output, label)    
    #     # compute the accuracy
    #     print(torch.argmax(output, dim=1))
    #     print(label)
    #     print(torch.argmax(output, dim=1) == label)
    #     print(torch.sum(torch.argmax(output, dim=1) == label).item())
    #     print(loss.item())
    #     print(len(valid_loader.dataset))
    #     print(len(valid_loader))
    # print(img.shape)
    # print(label.shape)
    # print(output.shape)
    # print(loss.item())
    # print(output)

    


    
    
    
