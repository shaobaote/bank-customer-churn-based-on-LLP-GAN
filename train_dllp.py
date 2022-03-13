import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch
import torch.autograd as autograd

#import libraries self-constructed
import datasets,losses,utils
from networks import Generator,Discriminator,DLLP_Classifier

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 10:
        lr = learning_rate * (0.5 ** (epoch // 5))  # i.e. 240,320
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_start(args):
    k=5#K折交叉验证
    X,Y=datasets.load_data()
    precision,recall,F1,acc,AUC=[],[],[],[],[]
    for i in range(k):
        print('第{}折交叉验证\n'.format(i+1))
        dataloader_train,dataloader_test = datasets.get_k_fold_data(k, i, X, Y,args.batch_size) 
        indicator=train(args,dataloader_train,dataloader_test)#返回值为列表
        precision.append(indicator[0])
        recall.append(indicator[1])
        F1.append(indicator[2])
        acc.append(indicator[3])
        AUC.append(indicator[4])
    print('5折交叉验证各指标均值和标准差:\n')
    print('precision={:.4f} {:.4f},recall={:.4f} {:.4f},F1={:.4f} {:.4f},acc={:.4f} {:.4f},AUC={:.4f} {:.4f}'.format(np.mean(precision),np.std(precision),np.mean(recall),np.std(recall),np.mean(F1),np.std(F1),np.mean(acc),np.std(acc),np.mean(AUC),np.std(AUC)))
    
        
def train(args,dataloader_train,dataloader_test):
    train_len,test_len=8000,2000
    batch_size=args.batch_size
    z_dim=args.z_dim
    lr=args.lr
    dataset_name=args.dataset_name
    n_epochs=args.n_epochs
    hy=args.hyperparameter
    beta1=args.beta1
    beta2=args.beta2
    betas=(beta1,beta2)
    method=args.method
    data_augmentation=args.data_augmentation
    eps=args.eps
    sample=args.sample
    
    #configure the environment and gpu
    #cuda=False
    cuda = True if torch.cuda.is_available() else False
    device=torch.device('cuda' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    #K折交叉验证
    
    #configure data
    
    bags=train_len//batch_size#the number of total bags of equal size on the training data

    # Initialize generator and discriminator
    D=DLLP_Classifier(15,logits=False).cuda()
    
    #configure optimizer
    optimizer_D = Adam(D.parameters(), lr=0.0001,betas=betas)
    
    xtick,y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[],[]
    
    for epoch in range(n_epochs):
        adjust_learning_rate(optimizer_D, epoch, lr)
        
        entropy_instance_level=[]
        cross_entropy_instance_level=[]
        Prop_loss=[]
        precision_list,recall_list,F1_list,acc_list=[],[],[],[]
        
        for i, (data, labels) in enumerate(dataloader_train):
            # Configure input
            real = data.cuda().to(torch.float32)
            real_prop=utils.get_categorical(labels,2,batch_size)

            optimizer_D.zero_grad()
            fake_prop_tmp=torch.mean(torch.sigmoid(D(real)),dim=0)
            fake_prop_tmp=torch.clamp(fake_prop_tmp, eps, 1 - eps)
            
            #prop_loss=torch.sum(losses.cross_entropy_loss(fake_prop,real_prop), dim=-1).mean()
            prop_loss=hy*torch.sum(-real_prop[1]*torch.log(fake_prop_tmp)-real_prop[0]*torch.log(1-fake_prop_tmp), dim=-1).mean()
            Prop_loss.append(prop_loss.item())
            
            if epoch>0:
                prop_loss.backward()
                optimizer_D.step()
        
        #evaluate on test
        entropy_instance_level,cross_entropy_instance_level,precision,recall,F1,acc,AUC=utils.eval_encoder_dllp(D,dataloader_test,dataset_name, method,train_len,test_len)
           
        xtick.append(epoch)
        y1.append(np.sum(entropy_instance_level)/test_len)
        y2.append(np.sum(cross_entropy_instance_level)/test_len)
        y4.append(acc)
        
        precision_list.append(precision);recall_list.append(recall);F1_list.append(F1);acc_list.append(acc)
        #save the figures
        utils.plot_loss('dllp',sample,z_dim,dataset_name, epoch+1, batch_size, hy, xtick, y1,y2,y4,precision,recall,F1,AUC)
    
        print('epoch={}\nprecision={:.4f} recall={:.4f} F1={:.4f} acc={:.4f} AUC={:.4f}'.format(epoch,precision_list[-1],recall_list[-1],F1_list[-1],acc_list[-1],AUC))
        print('Prop_loss={:.4f}\n'.format(Prop_loss[-1]))
    
    return [precision,recall,F1,acc,AUC]
