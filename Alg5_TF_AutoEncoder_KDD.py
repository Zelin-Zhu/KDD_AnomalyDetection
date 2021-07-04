# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 22:10:08 2021

@author: ASUS
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
#导入数据
data=np.load('./data/one_hot_data.npy')
X=data[:,0:-1] #特征数据
Y=data[:,-1]   #数据标签
normal_data=np.load('./data/data_normal.npy')
unnormal_data=np.load('./data/data_unnormal.npy')

#选择10000个normal_data作为训练集
#2500个normal_data作为测试集
#10000个UNnormal_data作为负样本测试集
num_train=10000
num_test=2500
num_unnormal_test=10000
X_train=normal_data[0:num_train,0:-1]
X_test=normal_data[num_train:num_train+num_test,0:-1]
unnormal_test=unnormal_data[0:num_unnormal_test,0:-1]

#设置encoder和decoder的各层单元数
layer_units=[41,30,20,10]

#构建AutoEncoder模型
AutoEncoder=tf.keras.Sequential()
##添加encoder部分
for i in range(len(layer_units)):
    AutoEncoder.add(layers.Dense(layer_units[i], activation='relu'))
##添加decoder部分
for i in range(len(layer_units)):
    AutoEncoder.add(layers.Dense(layer_units[len(layer_units)-1-i],activation='relu'))
    
#模型编译
AutoEncoder.compile(
                    optimizer=tf.keras.optimizers.RMSprop(),
                    loss='binary_crossentropy',
                    metrics=['acc'])

#模型训练
epochs=60
batch_size=512
history = AutoEncoder.fit(X_train,X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(unnormal_test, unnormal_test) 
                          )
AutoEncoder.evaluate(X_test,X_test)



