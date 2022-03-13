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
from networks import Generator,Discriminator

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
        
def zero_grad_all(optimizer_G,optimizer_D):
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

def train_start(args,dataloader_train,dataloader_test):
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
    
    bags=train_len//batch_size#the number of total bags of equal size on the training data

    # Initialize generator and discriminator
    G=Generator(z_dim,method).cuda()
    D=Discriminator(15,logits=False).cuda()
    
    #configure optimizer
    optimizer_G = Adam(G.parameters(), lr=0.0001,betas=betas)
    optimizer_D = Adam(D.parameters(), lr=0.0001,betas=betas)
    
    xtick,y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[],[]
    
    for epoch in range(n_epochs):
        adjust_learning_rate(optimizer_G, epoch, lr)
        adjust_learning_rate(optimizer_D, epoch, lr)
        
        entropy_instance_level=[]
        cross_entropy_instance_level=[]
        Prop_loss=[]
        G_loss=[]
        D_loss=[]
        GP_loss=[]
        Wasserstein_D=[]
        precision_list,recall_list,F1_list,acc_list=[],[],[],[]
        
        for i, (data, labels) in enumerate(dataloader_train):
            # Configure input
            real=data.cuda().to(torch.float32)
            if sample=='swiss_roll':
                random_z=Tensor(utils.swiss_roll(batch_size, z_dim)).cuda()
            elif sample=='gaussian_mixture':
                random_z=Tensor(utils.gaussian_mixture(batch_size, z_dim)).cuda()
            if sample=='normal':
                random_z=Tensor(utils.normal(batch_size, z_dim)).cuda()
            real_prop=utils.get_categorical(labels,2,batch_size)
            
            #######train discriminator#######
            zero_grad_all(optimizer_G,optimizer_D)
            fake = G(random_z)#生成假数据
            fake_prop_tmp=torch.mean(torch.sigmoid(D(real)),dim=0)#计算比例损失要用真数据
            fake_prop_tmp=torch.clamp(fake_prop_tmp, eps, 1 - eps)
            
            #fake_prop=torch.Tensor([fake_prop_tmp,1-fake_prop_tmp])
            #prop_loss=torch.sum(losses.cross_entropy_loss(fake_prop,real_prop), dim=-1).mean()
            
            prop_loss=torch.sum(-real_prop[1]*torch.log(fake_prop_tmp)-real_prop[0]*torch.log(1-fake_prop_tmp), dim=-1).mean()

            Prop_loss.append(prop_loss.item())
            #WGAN-GP
            gradient_penalty = 1*losses.cal_gradient_penalty(D, 'cuda', real, fake)
            d_loss_adv=1*( D(fake).mean()) + gradient_penalty
            with torch.no_grad():
                Wasserstein_D.append((D(real).mean() - D(fake).mean()).item())
                GP_loss.append(gradient_penalty.item())
            #GAN
            #d_loss=-0.02*torch.mean(torch.log(D(real) + eps)+torch.log(1-D(fake) + eps))
            
            D_loss.append(d_loss_adv.item())
            
            if epoch>0:
                d_loss=1*d_loss_adv + hy*prop_loss
                #d_loss= 0.1*prop_loss
                d_loss.backward()
                optimizer_D.step()
        
            #######train generator####### 
            zero_grad_all(optimizer_G,optimizer_D)
            fake = G(random_z)

            #WGAN-GP
            g_loss=-1*D(fake).mean()
            #GAN
            #g_loss_adv = - 0.2*torch.mean(torch.log(D(fake) + eps))
            G_loss.append(g_loss.item())
            
            if epoch>0:
                g_loss.backward()
                optimizer_G.step() 
        
        #evaluate on test
        entropy_instance_level,cross_entropy_instance_level,precision,recall,F1,acc,AUC=utils.eval_encoder(D,dataloader_test,dataset_name, method,train_len,test_len)
           
        xtick.append(epoch)
        y1.append(np.sum(entropy_instance_level)/test_len)
        y2.append(np.sum(cross_entropy_instance_level)/test_len)
        y4.append(acc)
        y5.append(np.mean(G_loss))
        y6.append(np.mean(D_loss))
        
        precision_list.append(precision);recall_list.append(recall);F1_list.append(F1);acc_list.append(acc)
        
        #save the figures
        utils.plot_loss('llp_gan',sample,z_dim,dataset_name, epoch+1, batch_size, hy, xtick, y1,y2,y4,y5,y6,precision,recall,F1,AUC)
    
        print('epoch={} \nprecision={:.4f} recall={:.4f} F1={:.4f} acc={:.4f} AUC={:.4f}'.format(epoch,precision_list[-1],recall_list[-1],F1_list[-1],acc_list[-1],AUC))
        print('Prop_loss={:.4f} G_loss={:.4f} D_loss={:.4f} GP_loss={:.4f} Wasserstein_D={:.4f}\n'.format(Prop_loss[-1],G_loss[-1],D_loss[-1],Wasserstein_D[-1],GP_loss[-1]))
    
        #save the model per epoch
        #torch.save(En.state_dict(),'./state_dict/'+ dataset_name + '/encoder '+str(hy)+' '+str(batch_size)+'.pth')
        #torch.save(De.state_dict(),'./state_dict/'+ dataset_name + '/decoder '+str(hy)+' '+str(batch_size)+'.pth')
