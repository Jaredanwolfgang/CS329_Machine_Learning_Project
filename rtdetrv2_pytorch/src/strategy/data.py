import numpy as np
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

AL_DATASET_BASE = os.path.join('.', 'dataset')

class Kitti_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.3680, 0.3873, 0.3760), (0.3034, 0.3105, 0.3137))
        ])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_kitti(handler):
    transform = transforms.Resize((256, 832))
    raw_train = datasets.KITTI(AL_DATASET_BASE, train=True, download=False, transform=transform)
    raw_test = datasets.KITTI(AL_DATASET_BASE, train=False, download=False, transform=transform)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler)
