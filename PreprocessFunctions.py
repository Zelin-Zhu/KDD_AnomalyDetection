# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:26:16 2021

@author: ASUS
"""
import numpy as np

def one_hot_data_split_normalize(data,train_pct,valid_pct,test_pct):
    if train_pct+valid_pct+test_pct!=1:
        print("input Error: Sum of train_pct,valid_pct,test_pct must equals 1")
        return 0
    np.random.seed(123)#设置随机种子，可复现
    np.random.shuffle(data)#先将数据shuffle一下
    train_num=int(len(data)*train_pct)
    valid_num=int(len(data)*valid_pct)
    #test_num=len(data)-valid_num-train_num
    
    train_data=data[0:train_num,:]
    valid_data=data[train_num:train_num+valid_num,:]
    test_data=data[train_num+valid_num:len(data),:]
    
    #计算连续值列的var和std 
    #80:117 连续值
    mean=np.mean(train_data[:,80:117],axis=0)
    std=np.std(train_data[:,80:117],axis=0)
    for i in range(len(std)): #如果某一列std==0则置1
        if std[i]==0:
            std[i]=1
    #归一化数据
    train_data[:,80:117]=(train_data[:,80:117]-mean)/std
    valid_data[:,80:117]=(valid_data[:,80:117]-mean)/std
    test_data[:,80:117]=(test_data[:,80:117]-mean)/std
    
    print("0-79为0-1变量，80-117列为连续变量，118列为标签")
    print("mean:",mean,"std:",std)
    return train_data,valid_data,test_data,mean,std

#train,valid,test=shuff_data=data_split(load_data, 0.5, 0.25, 0.25)

def one_hot_data_normalize(data,mean,std):
    data[:,80:117]=(data[:,80:117]-mean)/std
    return data

def factor_data_split_normalize(data,train_pct,valid_pct,test_pct):
    if train_pct+valid_pct+test_pct!=1:
        print("input Error: Sum of train_pct,valid_pct,test_pct must equals 1")
        return 0
    np.random.seed(123)#设置随机种子，可复现
    np.random.shuffle(data)#先将数据shuffle一下
    train_num=int(len(data)*train_pct)
    valid_num=int(len(data)*valid_pct)
    #test_num=len(data)-valid_num-train_num
    
    train_data=data[0:train_num,:]
    valid_data=data[train_num:train_num+valid_num,:]
    test_data=data[train_num+valid_num:len(data),:]
    
    #标准化,最后一列为标签列
    mean=np.mean(train_data[:,0:-1],axis=0)
    std =np.std (train_data[:,0:-1],axis=0)
    for i in range(len(std)): #如果某一列std==0则置1
        if std[i]==0:
            std[i]=1
    train_data[:,0:-1]=(train_data[:,0:-1]-mean)/std
    valid_data[:,0:-1]=(valid_data[:,0:-1]-mean)/std
    test_data[:,0:-1] =(test_data[:,0:-1]-mean)/std
    #print("mean:",mean,"std:",std)
    return train_data,valid_data,test_data,mean,std

def factor_data_normalize(data,mean,std):
    print("根据train数据的mean和std来归一化valid和test数据")
    #print("mean.shape:",mean.shape)
    #print("data[:,0:-1].shape:",data[:,0:-1].shape)    
    data[:,0:-1]=(data[:,0:-1]-mean)/std
    return data


