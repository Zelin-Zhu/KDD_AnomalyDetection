import numpy as np
from sklearn import svm

normal_data=np.load('./data/data_normal.npy')
unnormal_data=np.load('./data/data_unnormal.npy')


#选择normal的数据集子集作为训练集合
#选择unnormal数据集和剩余normal数据集作为测试集
import PreprocessFunctions as pf
train_pct=0.5
valid_pct=0.25
test_pct=0.25
#划分数据集并归一化
train_data,valid_data,test_data,mean,std=pf.factor_data_split_normalize(normal_data,train_pct,valid_pct,test_pct)
unnormal_data=pf.factor_data_normalize(unnormal_data, mean, std)


#从train_data数据集中选择一定数量的训练集
num_train=40000


train_data=train_data[0:num_train,0:-1]#去除最后一列的标签列
valid_data_normal=valid_data[:,0:-1]
valid_data_unnormal=unnormal_data[0:len(valid_data_normal),0:-1]#验证集正负样本数量相同
test_data_normal=test_data[:,0:-1]
test_data_unnormal=unnormal_data[len(valid_data_unnormal):len(unnormal_data),0:-1]

#构建分类器，训练分类器
clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
clf.fit(train_data)

##分类器预测
train_predict = clf.predict(train_data)
valid_normal_predict=clf.predict(valid_data_normal)
valid_unnormal_predict=clf.predict(valid_data_unnormal)
test_normal_predict=clf.predict(test_data_normal)
test_unnormal_predict=clf.predict(test_data_unnormal)



#正样本预测为1，负样本预测为-1


#计算预测准确率
def get_acc(prediction,labels):
    if len(prediction)!=len(labels):
        print("data size not match")#判断label和预测值的长度是否相同
    else:
        true_predict_num=0
        for i in range(len(prediction)):
            if labels[i]==prediction[i]:
                true_predict_num=true_predict_num+1
        return true_predict_num/len(prediction)    
    
    
train_labels=np.ones(len(train_data))  
valid_normal_labels   =  np.ones(len(valid_data_normal))
valid_unnormal_labels = -1*np.ones(len(valid_data_unnormal))
test_normal_labels    = np.ones(len(test_data_normal))
test_unnormal_labels  = -1*np.ones(len(test_data_unnormal))



train_acc=get_acc(train_predict, train_labels)
valid_normal_acc   = get_acc(valid_normal_predict,valid_normal_labels)
valid_unnormal_acc = get_acc(valid_unnormal_predict,valid_unnormal_labels)
test_normal_acc    = get_acc(test_normal_predict, test_normal_labels)
test_unnormal_acc  = get_acc(test_unnormal_predict, test_unnormal_labels)
print('train_acc:\t',train_acc)
print('valid_noraml_acc:\t',valid_normal_acc)
print('valid_unnoraml_acc:\t',valid_unnormal_acc)
print('test_noraml_acc:\t',test_normal_acc)
print('test_unnoraml_acc:\t',test_unnormal_acc)



# train_acc:	 0.89985
# valid_noraml_acc:	 0.8976109215017065
# valid_unnoraml_acc:	 0.9988486368682923
# test_noraml_acc:	 0.8993421052631579
# test_unnoraml_acc:	 0.9998791699783043
