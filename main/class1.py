# 数据操作
from multiprocessing import reduction
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
from tqdm import tqdm

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
    raw_x_test = test_data  ## 测试集无结果，不去除label

    if select_all:
        #`shape` 不是一个函数，而是 NumPy 数组（`numpy.ndarray`）的一个属性。NumPy 是一个用于科学计算的 Python 库，`shape` 属性用于获取数组的维度信息。
        feat_idx=list(range(raw_x_train.shape[1]))
        print("feat_idx" ,feat_idx)
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
        #`torch.FloatTensor` 是 PyTorch 中的一种数据类型，用于表示浮点型张量。它是 PyTorch 中最常用的张量类型之一，通常用于存储和处理连续数值数据。
        
       
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
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(), 
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
            #`nn.Sequential` 是一个有序容器，允许你将多个层按顺序组合在一起。输入数据会按顺序通过这些层。
            # nn.Linear` 是 PyTorch 中定义全连接层（或线性层）的模块。全连接层是神经网络的基本构建块之一，主要用于将输入数据进行线性变换
            # 在 PyTorch 中，当你定义一个线性层（例如 `nn.Linear`）时，该层会自动创建和初始化权重矩阵和偏置向量。这些参数存储在 `nn.Linear` 对象的属性中：
            # ReLU 激活函数，引入非线性变换

        # 前向传播    数据处理过程
        ## 假设 input_data 是输入张量
        ## model = MyModel(input_dim=10)
        ## output = model(input_data) 
 # 这里会自动调用 forward 方法
        def forward(self, x):
            x = self.layers(x)
            x = x.squeeze(1) # 去除维度为1的维度
            return x    
        
#参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
'seed':5201314,
'select_all' : True,
'valid_ratio':0.2,
'n_epochs' : 300, #循环次数
'batch_size' : 256,  #批次大小
'learning_rate' : 1e-5, #学习率
'early_stop': 40, #早停次数
'save_path' : './models/model.ckpt'
}

#训练


# step1. 定义损失函数


def trainer(train_loader, valid_loader,model,config,device):
    # mean表示平均值， 平方差的均值
    criterion = nn.MSELoss(reduction='mean')
    # 初始化优化器 PyTorch 中的随机梯度下降优化器（SGD），用于更新模型的参数以最小化损失函数。
    # model.parameters()`：获取模型中所有需要优化的参数。
    optimizer = torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=0.9)
    #数据可视化
    writer = SummaryWriter()

    # 模型保存路径
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    
    n_epochs =  config['n_epochs']
    #记录最小损失
    best_loss = math.inf
    # 记录训练步数
    step = 0
    # 记录早停次数
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train() # ？？
        train_loss = []
        train_bar = tqdm(train_loader,position = 0,leave = True)
        # train_pbar = tqdm() 进度条可视化  pip install tqdm
        #x是特征，y是标签
        for x,y in train_bar:
            # 数据传入模型
            x,y = x.to(device),y.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 模型预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred,y)
            loss.backward()
            #计算出的梯度会存储在每个模型参数的 `.grad` 属性中。这是反向传播计算的结果，用于之后的优化步骤。
            # 梯度更新
            optimizer.step()
            # 参数更新步数
            step += 1
            # 记录损失
            train_loss.append(loss.detach().item())
            # 数据可视化
            train_bar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_bar.set_postfix({'train_loss':loss.detach().item()})
            
            
        #平均损失    
        mean_train_loss = sum(train_loss)/len(train_loss)
        writer.add_scalar('Loss/train',mean_train_loss,step)
        
    # 模型验证
    model.eval()
    valid_loss = []
    for x,y in valid_loader:
        x,y = x.to(device),y.to(device)
        # 模型预测
        y_pred = model(x)
        # 计算损失
        loss = criterion(y_pred,y)
        # 记录损失
        valid_loss.append(loss.detach().item())
    #平均损失
    mean_valid_loss = sum(valid_loss)/len(valid_loss)

    if mean_valid_loss < best_loss:
        best_loss = mean_valid_loss
        # 保存模型
        torch.save(model.state_dict(),config['save_path'])
        print('Saving model with loss {:.3f}...'.format(best_loss))
        early_stop_count = 0
    else:
        early_stop_count += 1    

    if early_stop_count >= config['early_stop']:
        print('\nModel is not improving, so we halt the training session.')
        return
    


#设置随机种子

same_seed(config['seed'])

train_data= pd.read_csv('./covid.train.csv').values
test_data= pd.read_csv('./covid.test.csv').values

train_set ,valid_set =train_valid_split(train_data,config["valid_ratio"],config["seed"])
print("train_set size: {}".format(train_set.shape))
print("valid_set size: {}".format(valid_set.shape))

# 选择特征
raw_x_train,raw_x_valid,raw_x_test ,raw_y_train ,raw_y_valid =select_feature(train_set,valid_set,test_data,config["select_all"])
print("raw_x_train:", raw_x_train)
print("raw_x_valid:", raw_x_valid)
print("raw_x_test:", raw_x_test)
print("raw_y_train:", raw_y_train)
print("raw_y_valid:", raw_y_valid)
#准备dataset

train_dataset = COVID19Dataset(raw_x_train,raw_y_train,mode='train')
test_dataset = COVID19Dataset(raw_x_valid,raw_y_valid,mode='train')
valid_dataset = COVID19Dataset(raw_x_test,mode='test')

# 准备dataloader

train_dataloader =DataLoader(train_dataset,config['batch_size'],shuffle=True,drop_last=True)
valid_dataloader =DataLoader(train_dataset,config['batch_size'],shuffle=True,drop_last=True)
test_dataloader =DataLoader(train_dataset,config['batch_size'],shuffle=True,drop_last=True)

modle = MyModle(input_dim=raw_x_train.shape[1])

trainer(train_dataloader,valid_dataloader,modle,config,device)
