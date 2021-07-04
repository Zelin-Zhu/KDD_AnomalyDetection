# KDD数据集异常检测
## 数据集
* 采用有标注的10%数据集共49万多条数据，其中正常数据9万多条，异常数据39万多条
   * 采用正常数据的50%作为训练集
   * 正常数据的25%和相同数量的异常数据作为验证集
   * 正常数据的25%和剩余异常数据作为测试集
## 数据预处理
* one-hot 编码
    对离散值列进行one-hot编码
* 序列编码
    对离散值列进行序列编码
* 标准化   
    计算训练集中非0-1的列均值mean和方差std，各数据集数据x=(x-mean)/std
    * x_{trian}=(x_{train}-mean(train_data))/std(train_data)
    * x_{valid}=(x_{valid}-mean(train_data))/std(train_data)
    * x_{test}=(x_{test}-mean(train_data))/std(train_data)
## 算法1：Isolation Forest
 * train_acc:	         0.9137
 * valid_noraml_acc:	 0.9146757679180887
 * valid_unnoraml_acc: 0.9187877791027591
 * test_noraml_acc:	   0.9150082236842105
 * test_unnoraml_acc:	 0.994863381522136
## 算法2：OneClassSVM
 * train_acc:	         0.89985
 * valid_noraml_acc:	 0.8976109215017065
 * valid_unnoraml_acc: 0.9988486368682923
 * test_noraml_acc:	   0.8993421052631579
 * test_unnoraml_acc:	 0.9998791699783043
## 算法3：AutoEncoder
* 全连接网络模型
   * train_acc:       0.9791319722856144 
   * val_acc:         0.9455569719149636
   * test_acc_normal: 0.9792763157894737
   * test_acc_unnormal: 0.9934375872661267
   * divide1=         2.4918280440836575
* 卷积神经网络模型
  * train_acc:         0.9940993852669668
  * val_acc:           0.9394711953616514
  * test_acc_normal:   0.9951069078947369
  * test_acc_unnormal: 0.9341932850729276

