# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:09:26 2021

@author: ASUS
"""

#数据处理 
import numpy as np
import pandas as pd

####################################序列编码
####################################
data=pd.read_csv('KDDdata/kddcup.data_10_percent_corrected.csv',header=None)
#离散值下标
discrete_indexs=[1,2,3,6,11,20,21]

##离散列变量
protocol_types=data[1].unique()     #array(['tcp', 'udp', 'icmp'], dtype=object)
service_types=data[2].unique()      #len(service_types)=66
flag_types=data[3].unique()         #array(['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0','OTH', 'SH'], dtype=object)
land_types=data[6].unique()         #array([0, 1], dtype=int64)
logged_in_types=data[11].unique()   #array([1, 0], dtype=int64)
is_host_login_types=data[20].unique()       #array([0], dtype=int64)
is_guest_login_types=data[21].unique()      #array([0, 1], dtype=int64)

#将前三个离散值列和label列数字编码
#第41列为攻击类型normal为0 unnormal！=0
discrete_indexs_encode=[1,2,3,41]
for i in discrete_indexs_encode:
    data[i]=pd.factorize(data[i])[0]

#保存预处理数据
data=np.array(data)
np.save('.\data\data.npy',data)

#定义数据集分割和保准化函数
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
    train_data[:,0:-1]=(train_data[:,0:-1]-mean)/std
    valid_data[:,0:-1]=(valid_data[:,0:-1]-mean)/std
    test_data[:,0:-1] =(test_data[:,0:-1]-mean)/std
    print("mean:",mean,"std:",std)
    return train_data,valid_data,test_data,mean,std

def factor_data_normalize(data,mean,std):
    print("根据train数据的mean和std来归一化valid和test数据")
    data[:,0:-1]=(data[:,0:-1]-mean)/std
    return data

#分别保存正常数据和非正常数据
unnormal_index=np.squeeze(np.argwhere(data[:,41]>0))
unnormal_data=data[unnormal_index]
normal_index=np.squeeze(np.argwhere(data[:,41]==0))
normal_data=data[normal_index]
np.save('.\data\data_normal.npy',normal_data)
np.save('.\data\data_unnormal.npy',unnormal_data)

###################################################one-hot编码
###################################################
names=['duration','protocol_type','service','flag','src_bytes','dst_bytes', \
        'land','wrong_fragment','urgent','hot','num_failed_login',\
'logged_in','num_compromised','root_shell','su_attempted','num_root',\
'num_file_creations','num_shells','num_access_files','num_outbound_cmds',\
'is_host_login','is_guest_login','count','srv_count','serror_rate',\
'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',\
'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',\
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',\
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','types']

data=pd.read_csv('KDDdata/kddcup.data_10_percent_corrected.csv',header=None)
data.columns=names
#将离散列数据one-hot编码
protocol=pd.get_dummies(data['protocol_type'])
service=pd.get_dummies(data['service'])
flag=pd.get_dummies(data['flag'])
types=pd.factorize(data['types'])[0]
types=pd.DataFrame(types)
#删除离散列
data=data.drop(['protocol_type','service','flag','types'],axis=1)
#拼接离散值列和连续值列
new_data= pd.concat([protocol,service,flag,data,types], axis=1)
#new_data有119列      #0:79 为离散值      #80:117 连续值     # 118type

#保存数据
new_data=np.array(new_data)
np.save('.\data\one_hot_data.npy',new_data)
#导入数据load_data=np.load('./data/one_hot_data.npy')

#分别保存正常和非正常数据
unnormal_index=np.squeeze(np.argwhere(new_data[:,len(new_data[0])-1]>0))
new_unnormal_data=new_data[unnormal_index]
normal_index = np.squeeze(np.argwhere(new_data[:,len(new_data[0])-1]==0))
new_normal_data  =new_data[normal_index]
np.save('.\data\one_hot_data_normal.npy',new_normal_data)
np.save('.\data\one_hot_data_unnormal.npy',new_unnormal_data)


#定义一个数据预处理函数(分割数据集+标准化)
#输入data和train,valid,test 的比例 example(0.5:0.25:0.25)
#计算train数据集中离散列的mean 和std 
#将所有离散列的值减去mean再除以std

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


