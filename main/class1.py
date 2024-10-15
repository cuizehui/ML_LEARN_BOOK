# 数据操作
import numpy as np
import math 
# 读取，写入
import pandas as pd
import torch 
import os 
import csv
## pytorch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset,DataLoader,random_split
# 绘制图像
from   torch.utils.tensorboard import SummaryWriter


## 确定随机种子，保证每次结果相同
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    #GPU可用
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


## 划分数据集
def train_valid_split(data_set, valid_ratio,seed):
    #验证集大小
    valid_data_size =int(len(data_set)*valid_ratio)
    #训练集集大小
    train_data_size = len(data_set)-valid_data_size
    # 划分数据集 
    # 生成器，根据随机种子，随机划分数据集 generator=torch.Generator().manual_seed(seed)
    tran_data,valid_data =random_split(data_set,[train_data_size,valid_data_size],generator=torch.Generator().manual_seed(seed))
    return np.array(tran_data),np.array(valid_data)


## 选择特征



class COVID19Dataset(Dataset):
    def __init__(self,cfg,mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.data,self.scaler = self.load_data()
        self.features,self.targets = self.split_data(self.data)
        if mode == 'train':
            self.features = self.features[:int(len(self.features)*0.8)]

    
    def __getitem__(self, index) :
      
   