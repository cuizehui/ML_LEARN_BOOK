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
# 传入np数组 

def select_feature(train_data,valid_data,test_data,select_all=True):
    #挑选label
    #选择全部行和最后一列
    y_train= train_data[:,-1]
    y_valid= valid_data[:,-1]
    # 去掉label的数据
    raw_x_train = train_data[:,:-1]
    raw_x_valid = valid_data[:,:-1]
    raw_x_test = test_data[:,:-1]

    if select_all:
        #`shape` 不是一个函数，而是 NumPy 数组（`numpy.ndarray`）的一个属性。NumPy 是一个用于科学计算的 Python 库，`shape` 属性用于获取数组的维度信息。
        feat_idx=list(range(raw_x_train.shape[1]))
    else :
        feat_idx=[0,1,2,3,4]
    #返回训练集特征，返回验证集特征，返回测试集特征，返回训练集标签（预测结果），返回验证集标签（预测结果）
    return raw_x_train[:,feat_idx],raw_x_valid[:,feat_idx],raw_x_test[:,feat_idx] ,  y_train,y_valid


# 准备数据集
class COVID19Dataset(Dataset):
    # 初始化 传入特征和标签
    def __init__(self,features,targets=None,mode='train'):
        if(targets is None):
            self.targets = targets
        else:   
            self.targets = torch.FloatTensor(targets)
        self.features = torch.FloatTensor(features)

        
       
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if(self.targets == None):
            return self.features[idx]
        else:
            # 传入特征和标签
            return self.features[idx],self.targets[idx]
        


class MyModle(nn.Module):
        # 初始化
        def __init__(self, input_dim):
            super(MyModle, self).__init__()
            # 定义层
            self.layers = nn.Sequential{
                nn.Linear(input_dim, 16),
                nn.ReLU(), 
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            }
            #`nn.Sequential` 是一个有序容器，允许你将多个层按顺序组合在一起。输入数据会按顺序通过这些层。
            # nn.Linear` 是 PyTorch 中定义全连接层（或线性层）的模块。全连接层是神经网络的基本构建块之一，主要用于将输入数据进行线性变换
            # 在 PyTorch 中，当你定义一个线性层（例如 `nn.Linear`）时，该层会自动创建和初始化权重矩阵和偏置向量。这些参数存储在 `nn.Linear` 对象的属性中：
            # ReLU 激活函数，引入非线性变换

        # 前向传播    数据处理过程
        def forward(self, x):
            x = self.layers(x)
            x = x.squeeze(1) # 去除维度为1的维度
            return x    
        
#参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
'seed':1122408,
'select_all' : True,
'valid_ratio':0.2,
'n_epochs' : 3000, #循环次数
'batch_size' : 256,  #批次大小
'learning_rate' : 1e-5, #学习率
'early_stop': 400, #早停次数
'save_path' : './models/model.ckpt'
}