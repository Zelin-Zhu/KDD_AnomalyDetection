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
    * $x_{trian}=(x_{train}-mean(train_data))/std(train_data)$
    * x_{valid}=(x_{valid}-mean(train_data))/std(train_data)
    * x_{test}=(x_{test}-mean(train_data))/std(train_data)
## 算法1：Isolation Forest
## 算法2：OneClassSVM
## 算法3：AutoEncoder

