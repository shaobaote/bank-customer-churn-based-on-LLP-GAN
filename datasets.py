import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import Sampler, BatchSampler, RandomSampler
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch


# Configure data floder
os.makedirs("../dataset/UCI", exist_ok=True)
os.makedirs("./images/UCI", exist_ok=True)
os.makedirs("./index/UCI", exist_ok=True)
os.makedirs("./state_dict/UCI", exist_ok=True)

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

class BagSampler(Sampler):
    def __init__(self, bags, num_bags=-1):
        """
        params:
            bags: shape (num_bags, num_instances), the element of a bag
                  is the instance index of the dataset
            num_bags: int, -1 stands for using all bags
        """
        self.bags = bags
        if num_bags == -1:
            self.num_bags = len(bags)
        else:
            self.num_bags = num_bags
        assert 0 < self.num_bags <= len(bags)

    def __iter__(self):
        indices = torch.randperm(self.num_bags)
        for index in indices:
            yield self.bags[index]

    def __len__(self):
        return len(self.bags)

    
def load_data():
    df=pd.read_csv('D:/mac归档/研究生小论文和毕设/毕业论文/data/bank customer Churn.csv')
    #随机打乱数据
    np.random.seed(1)
    df=df.sample(frac=1).reset_index(drop=True)
    #训练结果，是否流失
    result_var='Exited'
    #分类型数据，需要预处理
    cat_names=['Gender','Geography']
    #数值型数据，可直接输入模型
    cont_names=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
    #标签
    Y=df['Exited'].astype('int')
    #类别变量变为独热编码，总的有4个类别变量
    one_hot=pd.DataFrame(columns=['Female','Male','France','Germany','Spain','HasCrCard_Yes','HasCrCard_No','IsActiveMember_Yes','IsActiveMember_No'],index=[i for i in range(10000)])
    one_hot.loc[df.Gender=='Female','Female']=1
    one_hot.loc[df.Gender=='Male','Female']=0
    one_hot.loc[df.Gender=='Male','Male']=1
    one_hot.loc[df.Gender=='Female','Male']=0
    one_hot.loc[df.Geography=='France','France']=1
    one_hot.loc[df.Geography!='France','France']=0
    one_hot.loc[df.Geography=='Germany','Germany']=1
    one_hot.loc[df.Geography!='Germany','Germany']=0
    one_hot.loc[df.Geography=='Spain','Spain']=1
    one_hot.loc[df.Geography!='Spain','Spain']=0
    one_hot.loc[df.HasCrCard==1,'HasCrCard_Yes']=1
    one_hot.loc[df.HasCrCard==0,'HasCrCard_Yes']=0
    one_hot.loc[df.HasCrCard==1,'HasCrCard_No']=0
    one_hot.loc[df.HasCrCard==0,'HasCrCard_No']=1
    one_hot.loc[df.IsActiveMember==1,'IsActiveMember_Yes']=1
    one_hot.loc[df.IsActiveMember==0,'IsActiveMember_Yes']=0
    one_hot.loc[df.IsActiveMember==1,'IsActiveMember_No']=0
    one_hot.loc[df.IsActiveMember==0,'IsActiveMember_No']=1
    #X为数据集，未区分训练集和测试集，用pandas提前划分；标准化处理非独热编码
    X=pd.concat([df[cont_names],one_hot],axis=1).astype('float')
    
    #归一化
    X=X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))#生成器最后一层是Tanh(-1,1)，把数据标准化到-1到1之间
    
    #标准化
    #X=pd.concat([df[cont_names].apply(lambda x: (x - np.mean(x)) / np.std(x)),one_hot],axis=1).astype('float')
    
    #print(X.loc[10,:])
    #print(Y.loc[10])
    return X,Y
    #X_.reset_index(drop=True, inplace=True);Y_test.reset_index(drop=True, inplace=True)


########k折划分############        
def get_k_fold_data(k, i, X, y,batch_size):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据   
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    X_train, Y_train = pd.DataFrame(), pd.Series()#要用series
    for j in range(k):
        ##idx 为每组 valid
        X_part = X.iloc[j * fold_size: (j + 1) * fold_size,:]
        
        Y_part=y[j * fold_size: (j + 1) * fold_size]
        if j == i: ###第i折作valid
            X_test, Y_test = X_part, Y_part
        #elif X_train is None:
            #X_train, Y_train = X_part, Y_part
        else:
            X_train = pd.concat([X_train, X_part], axis=0) #dim=0增加行数，竖着连接
            Y_train = pd.concat([Y_train, Y_part], axis=0)
    

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    Y_train.reset_index(drop=True, inplace=True)
    Y_test.reset_index(drop=True, inplace=True)
    
    train_set=MyDataset(X_train,Y_train)
    test_set=MyDataset(X_test,Y_test)
    
    indices = RandomSampler(range(8000), replacement=False)
    bags = list(BatchSampler(indices, batch_size=batch_size,drop_last=True))
    train_bag_sampler = BagSampler(bags)
    train_dataloader = DataLoader(train_set, batch_sampler=train_bag_sampler, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=200, shuffle=False, pin_memory=True, num_workers=4,drop_last=False)
    return train_dataloader, test_dataloader
