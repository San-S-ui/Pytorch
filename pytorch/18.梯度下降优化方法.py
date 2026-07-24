'''梯度下降相关介绍
概述：
梯度下降是结合 本次损失函数的导数(作为梯度) 基于学习率 来更新权重的。
公式：
W_{新} = W_{旧} - 学习率 * (本次的)梯度
存在的问题：
1. 遇到平缓区域，梯度下降(权重更新)可能会慢。
2. 可能会遇到 鞍点(梯度为0)
3. 可能会遇到 局部最小值。
解决思路：
从上述的 学习率 或者 梯度入手，进行优化，于是有了：动量法Momentum，自适应学习率AdaGrad，RMSProp，综合衡量：Adam

动量法Momentum：
动量法公式：
S_t = β * S_{t-1} +(1-β)G_t
解释：
S_t：本次的指数移动加权平均结果。
β：调节权重系数，越大，数据越平缓，历史指数移动加权平均 比重越大，本次梯度权重越小。
S_{t-1}：历史的指数移动加权平均结果。
G_t：本次计算出的梯度(不考虑历史梯度)。

## 加入动量法后的 梯度更新公式：
W_{新} = W_{旧} - 学习率 * S_t

'''
import torch
import torch.nn as nn
import torch.optim as optim

def demo1():
    #初始化权重
    w = torch.tensor([1.0],dtype=torch.float,requires_grad=True)
    #定义损失函数
    criterion = ((w**2)/2.0)
    #定义优化器 SGD是Polyak动量公式为 S_t = β * S_{t-1} +G_t
    optimizer = optim.SGD([w],lr=0.01,momentum=0.9)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')#w.grad是梯度 S0=0
#第二次更新
    criterion = ((w**2)/2.0)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')#w.grad是梯度  0.99-（0.9*1+0.1*0.99）*0.01
'''
已知条件：

初始 w0​=1.0，学习率 lr=0.01，动量 momentum=0.9
损失 L=w**2/2​，梯度 g=w

第一轮更新

计算当前梯度：g1​=w0​=1.0
更新动量速度：v1​=0.9×v0​+g1​=0.9×0+1.0=1.0
更新权重：w1​=w0​−lr×v1​=1.0−0.01×1.0=0.99

第二轮更新

计算当前梯度：g2​=w1​=0.99
更新动量速度：v2​=0.9×v1​+g2​=0.9×1.0+0.99=1.89
更新权重：w2​=0.99−0.01×1.89=0.9711
'''



'''
自适应学习率：AdaGrad
公式：
累计平方梯度：
S_t = S_{t-1} + Gt * Gt
解释：
S_t：累计平方梯度
S_{t-1} ：历史累计平方梯度。
G_t ：本次的梯度。

学习率：
学习率 = 学习率 / (sqrt(S_t) + 小常数)
#### 解释：
小常数：1e-10，目的：防止分母变为0

梯度下降公式：
W_{新} = W_{旧} - 调整后的学习率 * G_t

缺点：
可能会导致学习率过早，过量的降低，导致模型后期学习率太小，较难找到最优解。
'''

def demo2():
    #初始化权重
    w = torch.tensor([1.0],dtype=torch.float,requires_grad=True)
    #定义损失函数
    criterion = ((w**2)/2.0)
    #定义优化器
    optimizer = optim.Adagrad([w],lr=0.01)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')#w.grad是梯度 S0=0
#第二次更新
    criterion = ((w**2)/2.0)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

'''
自适应学习率：RMSProp
→ 可以看做是 对AdaGrad做的优化，加入 和权重系数。

公式：
指数加权平均 累计历史平方梯度：
 S_t = beta * S_{t-1} + (1 - beta) * Gt * Gt 
解释：
 S_t ：累计平方梯度
 S_{t-1} ：历史累计平方梯度。
 G_t ：本次的梯度。
 beta ：调和权重系数。

学习率：
学习率 = 学习率 /  (sqrt(S_t) + 小常数) 
解释：
小常数：1e-10，目的：防止分母变为0

梯度下降公式：
W_{新} = W_{旧} - 调整后的学习率 * G_t

优点：
RMSProp通过引入 衰减系数beta，控制历史梯度 对 历史梯度信息获取的多少。
'''
def demo3():
    #初始化权重
    w = torch.tensor([1.0],dtype=torch.float,requires_grad=True)
    #定义损失函数
    criterion = ((w**2)/2.0)
    #定义优化器
    optimizer = optim.RMSprop([w],lr=0.01,alpha=0.9)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')#w.grad是梯度 S0=0
#第二次更新
    criterion = ((w**2)/2.0)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
"""
自适应矩估计：Adam(Adaptive Moment Estimation)
思路：
即优化学习率，又优化梯度。

公式：
一阶矩：算均值。
M_t = beta_1 * M_{t-1} + (1 - beta_1) * G_t 充当：梯度
S_t = beta_2 * S_{t-1} + (1 - beta_2)*G_t* G_t 　充当：学习率

二阶矩：梯度的方差。
 hat{M_t} = M_t / (1 - beta_1 ^ t) 
 hat{S_t} = S_t / (1 - beta_2 ^ t) 

权重更新公式：
 W_{新} = W_{旧} - 学习率 / (sqrt(hat{S_t}) + 小常数) * hat{M_t} 

## 大白话翻译：
Adam = RMSProp + Momentum
"""

def demo4():
    #初始化权重
    w = torch.tensor([1.0],dtype=torch.float,requires_grad=True)
    #定义损失函数
    criterion = ((w**2)/2.0)
    #定义优化器
    optimizer = optim.Adam([w],lr=0.01,betas=(0.9,0.9))#betas(梯度衰减系数，学习率衰减系数)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')#w.grad是梯度 
#第二次更新
    criterion = ((w**2)/2.0)
    #梯度清零，反向传播，参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
if __name__=='__main__':
    #demo1()
    #demo2()
    #demo3()
    demo4()
'''
总结：如何选择梯度下降优化方法
简单任务和较小的模型：SGD，动量法
复杂任务或者有大量数据：Adam
需要处理稀疏数据或者文本数据：AdaGrad，RMSProp
'''
    
