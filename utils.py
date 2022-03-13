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

import datasets,losses
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def get_categorical(label, y_dim,batch_size):
    latent_y = torch.Tensor(np.eye(y_dim)[label].astype('float32'))
    if batch_size!=1:
        p=torch.mean(latent_y,dim=0).cuda()
    else:p=latent_y.cuda()
    return p

def gaussian_mixture(batch_size, n_dim, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z

def swiss_roll(batch_size, n_dim, n_labels=10, label_indices=None):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 5.0 * np.sqrt(uni)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z

def normal(batch_size, z_dim):
    z=np.random.normal(size=(batch_size,z_dim))
    return z

def eval_encoder_dllp(D,dataloader_test,dataset_name,method,train_len,test_len):
    err_epoch_test=0
    entropy_instance_level_epoch_test=[]
    cross_entropy_instance_level_epoch_test=[]
    epsilon=1e-8

    with torch.no_grad():
        TP,TN,FP,FN=0,0,0,0
        positive,negative=0,0
        predict,target=[],[]
        for i, (data, labels) in enumerate(dataloader_test):
            data=data.cuda().to(torch.float32)
            clf=D(data)
            fake=torch.sigmoid(clf)
            label_pred=fake.detach().cpu().data.numpy()
            #predict.extend(label_pred[0][:][1])
            target.extend(labels.cpu().data.numpy())
            cross_entropy_instance_level=0
            for j in range(data.shape[0]):
                if labels[j]==1:positive+=1
                else:negative+=1
                predict.append(np.clip(label_pred[j], epsilon, 1-epsilon))
                index=1 if label_pred[j]>=0.3 else 0
                if labels[j]==1 and index==1:
                    TP+=1
                elif labels[j]==0 and index==0:
                    TN+=1
                elif labels[j]==1 and index==0:
                    FN+=1
                elif labels[j]==0 and index==1:
                    FP+=1
                cross_entropy_instance_level+=-labels[j]*np.log(np.clip(label_pred[j], epsilon, 1-epsilon))-(1-labels[j])*np.log((1-np.clip(label_pred[j], epsilon, 1-epsilon)))
            cross_entropy_instance_level_epoch_test.append(cross_entropy_instance_level)
            
            #entropy instance-level on testing data
            ent_test=losses.cross_entropy_loss(fake,fake)
            ent_instance_level=torch.sum(ent_test,dim=1).sum()
            entropy_instance_level_epoch_test.append(ent_instance_level.item())
             
        precision=TP / (TP + FP +epsilon)#预测为正类的真正类比例
        recall=TP / (TP + FN +epsilon)#真正类预测为正的比例
        F1=2 * precision * recall / (precision + recall+epsilon )
        acc=(TP + TN) / (TP + TN + FP + FN +epsilon)
        AUC=roc_auc_score(target,predict)
        
        #print(positive,negative)
                
        return entropy_instance_level_epoch_test,cross_entropy_instance_level_epoch_test,precision,recall,F1,acc,AUC

def eval_encoder(D,dataloader_test,dataset_name,method,train_len,test_len):
    err_epoch_test=0
    entropy_instance_level_epoch_test=[]
    cross_entropy_instance_level_epoch_test=[]
    epsilon=1e-8

    with torch.no_grad():
        TP,TN,FP,FN=0,0,0,0
        positive,negative=0,0
        predict,target=[],[]
        for i, (data, labels) in enumerate(dataloader_test):
            data=data.cuda().to(torch.float32)
            clf,realfake=D(data)
            fake= F.softmax(clf,dim=1)
            label_pred=fake.detach().cpu().data.numpy()
            #predict.extend(label_pred[j][1])
            target.extend(labels.cpu().data.numpy())
            cross_entropy_instance_level=0
            for j in range(data.shape[0]):
                if labels[j]==1:positive+=1
                else:negative+=1
                predict.append(np.clip(label_pred[j][1], epsilon, 1-epsilon))
                index=1 if label_pred[j][1]>=0.3 else 0
                if labels[j]==1 and index==1:
                    TP+=1
                elif labels[j]==0 and index==0:
                    TN+=1
                elif labels[j]==1 and index==0:
                    FN+=1
                elif labels[j]==0 and index==1:
                    FP+=1
                cross_entropy_instance_level+=-labels[j]*np.log(np.clip(label_pred[j][1], epsilon, 1-epsilon))-(1-labels[j])*np.log((1-np.clip(label_pred[j][1], epsilon, 1-epsilon)))
                
            cross_entropy_instance_level_epoch_test.append(cross_entropy_instance_level)
            
            #entropy instance-level on testing data
            ent_test=losses.cross_entropy_loss(fake,fake)
            ent_instance_level=torch.sum(ent_test,dim=1).sum()
            entropy_instance_level_epoch_test.append(ent_instance_level.item())
             
        precision=TP / (TP + FP +epsilon)#预测为正类的真正类比例
        recall=TP / (TP + FN +epsilon)#真正类预测为正的比例
        F1=2 * precision * recall / (precision + recall+epsilon )
        acc=(TP + TN) / (TP + TN + FP + FN +epsilon)
        AUC=roc_auc_score(target,predict)
        
        #print(positive,negative)
                
        return entropy_instance_level_epoch_test,cross_entropy_instance_level_epoch_test,precision,recall,F1,acc,AUC

def plot_loss(algorithm,sample,z_dim, dataset_name, count, batch_size, hy, xtick,y1,y2,y4,precision,recall,F1,AUC):
    #save the data of index info
    if algorithm!='':
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+'.npy',y1)
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' cross_entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+'.npy',y2)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' acc_test '+str(hy)+ '_'+str(batch_size)+'.npy',y4)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' acc_test '+str(hy)+ '_'+str(batch_size)+'.npy',y4)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' precision '+str(hy)+ '_'+str(batch_size)+'.npy',precision)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' recall '+str(hy)+ '_'+str(batch_size)+'.npy',recall)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' F1 '+str(hy)+ '_'+str(batch_size)+'.npy',F1)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' AUC '+str(hy)+ '_'+str(batch_size)+'.npy',AUC)
    
        plt.clf()
        plt.figure(figsize=(5,8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.45)
    
        plt.subplot(511)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(dataset_name +' '+sample+''+str(z_dim)+' bagsize=' + str(batch_size)+' hy='+ str(hy) +' entropy instance-level test',fontsize=10)
        plt.plot(xtick[:count],y1[:count],color='red')
        
        plt.subplot(512)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(' cross entropy instance-level test(green)',fontsize=10)
        plt.plot(xtick[:count],y2[:count],color='green')
    
        plt.subplot(513)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title('acc test',fontsize=10)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.ylim([0,1])
        plt.plot(xtick[:count],y4[:count],color='red')
        
        plt.savefig('./images/'+ dataset_name + ' ' +algorithm + ' '+sample+ ' '+str(z_dim)+' loss bagsize=%d hy=%.2f.png'%(batch_size, hy ))
        plt.close('all')
        
def plot_all_loss(dataset_name,method,sample,z_dim,batch_size,hy,iteration,Prop_loss,fake_clf_loss,G_loss,D_loss,GP_loss,Wasserstein_D):
    xtick=list(np.arange(iteration))
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.45)
    
    plt.subplot(321)
    ax = plt.gca() 
    ax.tick_params(labelright = True)
    ax.grid()
    plt.title('Prop_loss',fontsize=10)
    plt.plot(xtick,Prop_loss,color='red',lw=0.5)
    
    plt.subplot(322)
    ax = plt.gca() 
    ax.tick_params(labelright = True)
    ax.grid()
    plt.title('G_loss(red) D_loss(green)',fontsize=10)
    plt.plot(xtick,G_loss,color='red',lw=0.5)
    plt.plot(xtick,D_loss,color='green',lw=0.5)
    
    plt.subplot(323)
    ax = plt.gca() 
    ax.tick_params(labelright = True)
    ax.grid()
    plt.title('GP_loss',fontsize=10)
    plt.plot(xtick,GP_loss,color='red',lw=0.5)
    
    plt.subplot(324)
    ax = plt.gca() 
    ax.tick_params(labelright = True)
    ax.grid()
    plt.title('Wasserstein_D',fontsize=10)
    plt.plot(xtick,Wasserstein_D,color='red',lw=0.5)
    
    plt.subplot(325)
    ax = plt.gca() 
    ax.tick_params(labelright = True)
    ax.grid()
    plt.title('fake_clf_loss',fontsize=10)
    plt.plot(xtick,fake_clf_loss,color='red',lw=0.5)
    
    plt.savefig('./images/loss/'+ dataset_name + ' ' +method + ' '+sample+ ' '+str(z_dim)+' loss bagsize=%d hy=%.2f.png'%(batch_size, hy ))
    plt.close('all')
    