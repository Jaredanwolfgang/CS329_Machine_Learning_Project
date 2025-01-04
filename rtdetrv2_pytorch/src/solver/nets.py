import numpy as np
import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
from typing import Iterable

from ._solver import BaseSolver
from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

class Net:
    def __init__(self, model, device):
        self.net = model
        self.device = device    

    def predict(self, data, loader):
        self.net.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = dist_utils.wrap_loader(loader, data, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.net(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    def predict_prob(self, data, loader):
        self.net.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = dist_utils.wrap_loader(loader, data, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.net(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def predict_prob_dropout(self, data, loader, n_drop=10):
        self.net.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = dist_utils.wrap_loader(loader, data, shuffle=False)
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.net(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, loader, n_drop=10):
        self.net.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = dist_utils.wrap_loader(loader, data, shuffle=False)
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.net(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_embeddings(self, data):
        self.net.eval()
        embeddings = torch.zeros([len(data), self.net.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.net(x)
                embeddings[idxs] = e1.cpu()
        return embeddings