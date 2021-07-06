# OneClassSVM

![image-20210706144245043](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210706144245043.png)

* 算法思想

  * 找到一个超球面，球面中心为a, 半径为R 使得所有正常样本在球面内，同时球面尽可能的小

  此时目标函数与约束条件为：

  <img src="C:\Users\ASUS\Desktop\图片1.png" alt="图片1" style="zoom:25%;" />

  * 由于一般情况下不能通过球面将正常样本和非正常样本完全分开，因此如果球面内包含所有正常样本，则可能包含更多的非正常样本。因此给样本点 $x_i $引入松弛条件 $\xi_i$ ,允许部分正常样本点在球面之外，同时对松弛距离在目标函数加入惩罚项，C为惩罚系数,C>0 ,根据需要人为设定。

    此时目标函数与约束条件为：

  ![image-20210706144457776](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210706144457776.png)

* 求解步骤

  1. 通过拉格朗日乘子法转化将条件极值转化为无约束极值

     拉格朗日函数如下：

     <img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210706145734543.png" alt="image-20210706145734543" style="zoom:80%;" />

​                $\alpha_i>=0$ ，$\gamma_i>=0$ ,对拉格朗日函数求最大值的条件等价于原条件极值的条件。此时原问题转化为先对拉格朗   日函数求最大值再求最小值即：
$$
\mathop{min}\limits_{R, a, \xi}\mathop{max}\limits_{\alpha,\gamma}L
$$


   2. 转化为对偶问题求解
      $$
      \mathop{min}\limits_{R, a, \xi}\mathop{max}\limits_{\alpha,\gamma}L=\mathop{max}\limits_{\alpha,\gamma}\mathop{min}\limits_{R, a, \xi}L
      $$
      根据  $\mathop{min}\limits_{R, a, \xi}L$ 求偏导

      <img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210706152311926.png" alt="image-20210706152311926" style="zoom: 80%;" />

将等式代入$L$得到：

<img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210706152537006.png" alt="image-20210706152537006" style="zoom:80%;" />

此时目标函数和约束条件如下：约束条件由求偏导的第三条等式可得

​                                                     <img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20210706152643755.png" alt="image-20210706152643755" style="zoom:80%;" />

  此时仅有 $\alpha$ 未知 可根据二次规划求解 $\alpha$，然后根据  $\alpha$ 和求偏导第三条等式得到 $\gamma$  ,根据 $\alpha$ 和第二条求偏导等式得到 $a $

根据KTT条件即原来的约束条件乘以拉格朗日乘数等于零即


$$
\gamma_i \xi_i=0 
\\ ||x_i-a||^{2}-R^2-\xi_i=0
$$
分别得到 $\xi_i$ 和 $R$

* 最终结果即为超球面$(x-a)^2<=R$

