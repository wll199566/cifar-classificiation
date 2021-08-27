# CNN for CIFAR-10 dataset
This repository is to practice my PyTorch for image classficiation problem.

The dataset is from https://www.cs.toronto.edu/~kriz/cifar.html

Downloaded data needs to be stored in `./data` folder

Files to execute:

1. `EDA.ipynb`: explore the raw data
2. `preprocess.ipynb`: preprocess the raw data
3. `sanity_checks.py`: sanity checks for the implementation
4. `hyper_tune.py`: tune hyperparameters
5. `train.py`: train the model based on best hyperparameters searched in step 4

The implemeted model here is based on the ResNet paper:
Deep Residual Learning for Image Recognition [He et al. 2015]
