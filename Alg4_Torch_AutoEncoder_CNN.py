# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:00:19 2021

@author: ASUS
"""

import torch as torch 
import torch.nn as nn
import numpy as np

#导入数据
normal_data=np.load('./data/one_hot_data_normal.npy')
unnormal_data=np.load('./data/one_hot_data_unnormal.npy')
#data有119列      #0:79 为离散值      #80:117 连续值     # 118type

##数据集分割
import PreprocessFunctions as pf
train_pct=0.5
valid_pct=0.25
test_pct=0.25
train_data,valid_data,test_data,mean,std=pf.one_hot_data_split_normalize(normal_data, train_pct, valid_pct, test_pct)
unnormal_data=pf.one_hot_data_normalize(unnormal_data, mean, std)




##定义一个对数据增加维度并转换为tensor的函数
def fix_data(data):
    data=data[:,0:-1]
    data=np.expand_dims(data,axis=1)
    data=torch.tensor(data)
    return data

train_data=fix_data(train_data)
valid_data_normal=fix_data(valid_data)
valid_data_unnormal=fix_data(unnormal_data[0:len(valid_data_normal)])
test_data_noraml=fix_data(test_data)
test_data_unnormal=fix_data(unnormal_data[len(valid_data_normal):len(unnormal_data)])
##定义模型

class AutoEncoder_CNN(nn.Module):
    def __init__(self):
        super(AutoEncoder_CNN,self).__init__()
        self.encoder=nn.Sequential(
                                   nn.Conv1d(1, 32, 9,stride=1,padding=8), #126
                                   nn.Sigmoid(),
                                   nn.MaxPool1d(2,stride=2),                #63
                                   nn.Conv1d(32,16,kernel_size=8,stride=1,padding=7),#70
                                   nn.Sigmoid(),
                                   nn.MaxPool1d(2,stride=2),                #35
                                   nn.Conv1d(16,8,kernel_size=8,stride=1,padding=7),#42
                                   nn.Sigmoid(),
                                   nn.MaxPool1d(2,stride=2), #21
                                   )
        self.decoder=nn.Sequential(nn.ConvTranspose1d(8, 16,  kernel_size=10,stride=2), #52
                                   nn.Sigmoid(),                
                                   nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=11,stride=2),#114
                                   nn.Sigmoid(),              
                                   nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=10,stride=1)#118
                            
                                   )
    def forward(self,x):
        
       #print("x.shape:",x.shape)
        #x = x.permute(0,2,1)
        y=self.encoder(x)
        #print("y.shape:",y.shape)
        z=self.decoder(y)
        #print('z.shape:',z.shape)
        return z



# ## 训练模型参数定义
#实例化模型
model=AutoEncoder_CNN()
#定义损失函数，学习率，梯度优化函数，batch_size
loss_fun=nn.MSELoss()
learning_rate=0.001
optimizer=torch.optim.Adam(model.parameters(),learning_rate)
batch_size=256
# 用dataloader进行batch训练
train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True)
###############训练模型
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
        

##计算loss值

def get_loss(data,model):
    losses=[]
    for i in range(len(data)):
        sample=torch.tensor(np.expand_dims(data[i],axis=0))
        loss=loss_fun(model(sample),sample)
        losses.append(loss.detach().numpy().item())
    return losses

loss_val_normal=get_loss(valid_data_normal,model)
loss_val_unnormal=get_loss(valid_data_unnormal,model)
loss_test_normal=get_loss(test_data_noraml, model)
loss_test_unnormal=get_loss(test_data_unnormal, model)
loss_train=get_loss(train_data, model)

##计算正常数据和非正常数据的loss分界值
def get_divide_value(loss_val_normal,loss_val_unnormal):
    ###第一次搜索
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
    return divide1

divide=get_divide_value(loss_val_normal, loss_val_unnormal)


#测试集准确率
test_acc_normal=np.sum(np.array(loss_test_normal)<=divide)/len(loss_test_normal)
test_acc_unnormal=np.sum(np.array(loss_test_unnormal)>divide)/len(loss_test_unnormal)
print("test_acc_normal:",test_acc_normal)
print("test_acc_unnormal:",test_acc_unnormal)

# test_acc_normal: 0.9951069078947369
# test_acc_unnormal: 0.9341932850729276

#训练集和验证集准确率
train_acc=np.sum(np.array(loss_train)<=divide)/len(loss_train)
val_acc=(np.sum(np.array(loss_val_normal)<=divide)+\
        np.sum(np.array(loss_val_unnormal)>divide))/(2*len(loss_val_normal))
print("train_acc:",train_acc)
print("val_acc:",val_acc)

# train_acc: 0.9940993852669668
# val_acc: 0.9394711953616514



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
            }, file_path+'KDD_AutoEncoder_CNN_test_acc_0.93_0.99.pt')

#模型导入
load_model = AutoEncoder_CNN()
load_optimizer = nn.MSELoss()

checkpoint = torch.load(file_path+'KDD_AutoEncoder_CNN_test_acc_0.93_0.99.pt')
load_model.load_state_dict(checkpoint['model_state_dict'])
#load_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
load_epoch = checkpoint['epoch']
load_loss = checkpoint['loss']
load_val_acc=checkpoint['val_acc']
load_train_acc=checkpoint['train_acc']
load_test_acc_normal=checkpoint['test_acc_normal']
load_test_acc_unnormal=checkpoint['test_acc_unnormal']
model.eval()















