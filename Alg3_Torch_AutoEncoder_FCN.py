#!/usr/bin/env python
# coding: utf-8

# # KDD异常检测
# * 导入数据已经对离散值数据列进行顺序编码,所有列已经归一化
# * 数据为KDD数据的10%（data_10_percent_corrected） 共49万多条，其中正常数据9万7千多条，非正常数据39万多条
# * 采用全连接神经网络自编码器模型，利用正常数据进行训练
# * 我们利用正常数据的3/4作为训练集，正常数据的1/4和非正常数据作为测试集

# In[1]:

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
#get_ipython().run_line_magic('matplotlib', 'inline')


# ##  导入数据

# In[2]:


normal_data=np.load('./data/data_normal.npy')
unnormal_data=np.load('./data/data_unnormal.npy')


# In[3]:





# ## 构建训练数据和测试数据集
# * 正常数据集的1/2作为训练集 train_data
# * 正常数据集的1/4作为验证集 valid_data_normal
# * 正常数据集的1/4作为测试集 test_data_normal
# 
# * 非正常数据集的len(valid_data_normal)样本作为非正常验证集 valid_data_unnormal
# * 非正常数据的其余样本作为 test_data_unnormal
import PreprocessFunctions as pf
train_pct=0.5
valid_pct=0.25
test_pct=0.25
train_data,valid_data,test_data,mean,std=pf.factor_data_split_normalize(normal_data,train_pct,valid_pct,test_pct)
unnormal_data=pf.factor_data_normalize(unnormal_data, mean, std)

train_data=torch.tensor(train_data[:,0:-1])
valid_data_normal=torch.tensor(valid_data[:,0:-1])
test_data_normal=torch.tensor(test_data[:,0:-1])
unnormal_data=unnormal_data[:,0:-1]
valid_data_unnormal=torch.tensor(unnormal_data[0:len(valid_data_normal)])
test_data_unnormal=torch.tensor(unnormal_data[len(valid_data_normal):len(unnormal_data)])

unnormal_data=torch.tensor(unnormal_data)

# ## 全连接自编码器
# * 每条数据经过编码后有41个特征
# * 编码器：
#   * 第一层32个单元
#   * 第二层16个单元
#   * 第三层8个单元
# * 解码器：
#   * 第一层16个单元
#   * 第二层32个单元
#   * 第三层41个单元
# * 中间层用ReLU激活，
# * 中间层可添加Sigmoid，训练速度可能加快
# * 输出加sigmoid 匹配输入数据的范围

# In[5]:


class AutoEncoder_FCN(nn.Module):
    def __init__(self):
        super(AutoEncoder_FCN,self).__init__()
        self.encoder=nn.Sequential(
                                   nn.Linear(41,32),
                                   nn.ReLU(),
                                   nn.Linear(32,16),
                                   nn.ReLU(),
                                   nn.Linear(16,8),
                                   nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(8,16),
                                   nn.ReLU(),
                                   nn.Linear(16,32),
                                   nn.ReLU(),
                                   nn.Linear(32,41),
                                   )
    def forward(self,x):
        y=self.encoder(x)
        z=self.decoder(y)
        return z
        


# ## 训练模型参数定义
#实例化模型
model=AutoEncoder_FCN()
#定义损失函数，学习率，梯度优化函数，batch_size
loss_fun=nn.MSELoss()
learning_rate=0.001
optimizer=torch.optim.Adam(model.parameters(),learning_rate)
batch_size=256
# 用dataloader进行batch训练
train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True)


# ## 训练模型

# In[7]:


epochs=1
losses=[]
for epoch in range(epochs):
    for index,data in enumerate(train_loader):
        model.double()
        model.train()
        optimizer.zero_grad()    #梯度清零
        output=model(data)       #正向传播
        loss=loss_fun(output,data)#计算损失
        loss.backward()          #反向传播
        optimizer.step()
        print('eopch:',epoch+1,"batch:",index+1,"/",len(train_loader),"batch_loss:",loss.detach().numpy())
    losses.append(loss.data) 
        
    


####计算各个集合的损失值
loss_train=[]
for i in range(len(train_data)):
    loss=loss_fun(model(train_data[i]),train_data[i])
    loss_train.append(loss.detach().numpy().item())


loss_val_normal=[]
for i in range(len(valid_data_normal)):
    loss=loss_fun(model(valid_data_normal[i]),valid_data_normal[i])
    loss_val_normal.append(loss.detach().numpy().item())


loss_test_normal=[]
for i in range(len(test_data_normal)):
    loss=loss_fun(model(test_data_normal[i]),test_data_normal[i])
    loss_test_normal.append(loss.detach().numpy().item())

loss_val_unnormal=[]
for i in range(len(valid_data_unnormal)):
    loss=loss_fun(model(valid_data_unnormal[i]),valid_data_unnormal[i])
    loss_val_unnormal.append(loss.detach().numpy().item())
    
loss_test_unnormal=[]
for i in range(len(test_data_unnormal)):
    loss=loss_fun(model(test_data_unnormal[i]),test_data_unnormal[i])
    loss_test_unnormal.append(loss.detach().numpy().item())
    

#找到一个loss分界值p ,当loss值>p时为非正常样本，否则为正常样本
#p满足 使得count(val_loss_normal<=p)+count(val_loss_unnoraml>p) 最大
#从min(loss_val_unnormal)和max(loss_val_normal)之间找，从小到大,间隔100，step=(max-min/100)
#第一次找到p1
#第二次从p1-step,p1+step 之间继续分隔100各间隔 找到p2 以此类推
left=min(loss_val_unnormal)
right=max(loss_val_normal)
step=(right-left)/100 
most_count=0
divide=left
for i in range(101):
    p=left+i*step
    true_count=np.sum(np.array(loss_val_normal)<=p)+np.sum(np.array(loss_val_unnormal)>p)
    if true_count>=most_count:
        most_count=true_count
        divide=p
   

#第二次搜索
left1=divide-step
right1=divide+step
step1=(right1-left1)/100
most_count1=0
divide1=divide
for i in range(101):
    p=left1+i*step1
    true_count=np.sum(np.array(loss_val_normal)<=p)+np.sum(np.array(loss_val_unnormal)>p)
    if true_count>=most_count1:
        divide1=p
        most_count1=true_count
    


val_acc=most_count1/2/len(loss_val_normal)
train_acc=np.sum(np.array(loss_train)<=divide1)/len(loss_train)
print("train_acc:",train_acc)
print("val_acc:",val_acc)


#测试集准确率
test_acc_normal=np.sum(np.array(loss_test_normal)<=divide1)/len(loss_test_normal)
test_acc_unnormal=np.sum(np.array(loss_test_unnormal)>divide1)/len(loss_test_unnormal)
print("test_acc_normal:",test_acc_normal)
print("test_acc_unnormal:",test_acc_unnormal)



# train_acc: 0.9791319722856144
# val_acc: 0.9455569719149636
# test_acc_normal: 0.9792763157894737
# test_acc_unnormal: 0.9934375872661267
#divide1=2.4918280440836575

# 模型保存
file_path='./models/'
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_acc': train_acc,
            'val_acc':val_acc,
            'test_acc_normal':test_acc_normal,
            'test_acc_unnormal':test_acc_unnormal
            }, file_path+'KDD_AutoEncoder_FCN_test_acc_0.98_0.99.pt')

#模型导入
load_model = AutoEncoder_FCN()
load_optimizer = nn.MSELoss()

checkpoint = torch.load(file_path+'KDD_AutoEncoder_FCN_test_acc_0.98_0.99.pt')
load_model.load_state_dict(checkpoint['model_state_dict'])
#load_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
load_epoch = checkpoint['epoch']
load_loss = checkpoint['loss']
load_val_acc=checkpoint['val_acc']
load_train_acc=checkpoint['train_acc']
load_test_acc_normal=checkpoint['test_acc_normal']
load_test_acc_unnormal=checkpoint['test_acc_unnormal']
model.eval()



# In[ ]:




