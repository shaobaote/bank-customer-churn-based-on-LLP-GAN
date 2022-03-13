import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    def __init__(self,z_dim, method, output_dim=15):
        super(Generator,self).__init__()
        self.input_dim=z_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, output_dim), nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, X):
        out = self.net(X)
        out=out.view(out.shape[0],-1)
        
        return out
    
class Discriminator(nn.Module):
    def __init__(self,input_dim,logits):
        super(Discriminator, self).__init__()
        self.out_net= nn.Sequential(
            nn.Linear(input_dim, 1000),nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            
        )
        self.clf=nn.Linear(1000,2)#,nn.Softmax(dim=1)
        self.realfake=nn.Linear(1000,1)
        self.apply(weights_init)

    def forward(self, z):
        out=self.out_net(z)
        clf=self.clf(out)
        realfake=self.realfake(out)
        return clf,realfake
        #return torch.sigmoid(out)

class DLLP_Classifier(nn.Module):
    def __init__(self,input_dim,logits):
        super(DLLP_Classifier, self).__init__()
        self.out_net= nn.Sequential(
            nn.Linear(input_dim, 100),nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100,1),
        )
   
        self.apply(weights_init)

    def forward(self, z):
        out=self.out_net(z)
        return out
        
#逻辑回归
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(15, 1)

    def forward(self, x):
        x = self.features(x)
        return x