import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch

#import libraries self-constructed
import networks,datasets,losses,utils,train_dllp

import warnings
warnings.filterwarnings("ignore")

def get_args():
    batch_size=1
    method='dllp'
    sample='normal'
    #sample='normal'
    #sample='swiss_roll'
    data_augmentation=False
    dataset_name='UCI'
    hy=10

    z_dim= 2
    
    parser = argparse.ArgumentParser("Learning from Label Proportions with Adversarial Autoencoder")

    # basic arguments
    parser.add_argument("--sample", type=str,default=sample)
    parser.add_argument("--dataset_name", type=str,default=dataset_name)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--method", type=str,default=method)
    parser.add_argument("--data_augmentation", type=str,default=data_augmentation)
    #parser.add_argument("--model_name", type=str, default="wrn28-2")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--metric", type=str, default="ce")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--z_dim", default=z_dim, type=int)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    
    # coefficient for balance fm-loss and prop-loss, hy is for fm-loss
    parser.add_argument("--hyperparameter", default=hy, type=int)

    return parser.parse_args()

def main(args):
    train_dllp.train_start(args)

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed(args.seed)
    main(args)

