"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import torch
import torch.utils.data
import torchvision
torchvision.disable_beta_transforms_warning()
import numpy as np
from PIL import Image
from torch.datasets import CocoDetection

from ._dataset import DetDataset
from ...core import register

__all__ = ['KittiDetection']

@register()
class KittiData:
    """
    Kitti is the dataset for Kitti object detection. Kitti dataset loads
    images and target from root and annotation file. The annotation file
    is in coco format.
    """
    def __init__(self, root, annTrainFile, annTestFile, train_transform=None, val_transform=None):
        self.train = CocoDetection(os.path.join(root, 'train2017'), annTrainFile, transforms=train_transform)
        self.val = CocoDetection(os.path.join(root, 'val2017'), annTestFile, transforms=val_transform)
        train_image_ids = self.train.getImgIds()
        self.X_train = self.train.loadImgs(train_image_ids)
        self.Y_train = [self.train.loadAnns(self.train.getAnnIds(imgIds=img['id'])) for img in self.X_train]
        
        val_image_ids = self.val.getImgIds()
        self.X_val = self.val.loadImgs(val_image_ids)
        self.Y_val = [self.val.loadAnns(self.val.getAnnIds(imgIds=img['id'])) for img in self.X_val]
        
        self.handler = KittiDetection
        self.n_pool = len(self.X_train)
        self.n_test = len(self.X_val)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
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
        return self.handler(self.X_val, self.Y_val)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_val==preds).sum().item() / self.n_test
    
    def extra_repr(self) -> str:
        s = "Number of samples: {n_pool}\n".format(n_pool=self.n_pool)
        s += "Number of test samples: {n_test}\n".format(n_test=self.n_test)
        return s 
        
    @property
    def categories(self, ):
        return self.train.dataset['categories']
    
    @property
    def category2name(self, ):
        return {c['id']: c['name'] for c in self.categories}
    
    @property
    def category2label(self, ):
        return {k: i for i, k in self.category2name.items()}
    
    @property
    def label2category(self, ):
        return {v: k for k, v in self.category2label.items()}
        

@register()
class KittiDetection(DetDataset):
    """
    Class that implements the Datasets, which can be formal input to DataLoader.
    """
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self._epoch = -1

    def __getitem__(self, index):
        x, y = self.load_item(index)
        if self.transforms is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index
    
    def load_item(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y
    
    def __len__(self):
        if(len(self.X) != len(self.Y)):
            raise ValueError("The number of images and annotations should be the same.")
        return len(self.X)
