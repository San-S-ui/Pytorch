'''概述：
损失函数也叫成本函数，目标函数，代价函数，误差函数，就是用来衡量 模型好坏 (模型拟合情况) 的。

分类：
分类问题：
多分类交叉熵损失：CrossEntropyLoss二分类交叉熵损失：BCELoss
回归问题：
MAE：Mean Absolute Error，平均绝对误差。MSE：Mean Squared Error，均方误差。Smooth L1：结合上述两个的特点做的升级，优化。

多分类交叉熵损失：CrossEntropyLoss
设计思路：
Loss=−∑ylog(S(f(x)))
简单记忆：
x：样本  f (x):加权求和    S (f (x))：Softmax ()处理后的概率   y：样本 x 属于某一个类别的 真实概率。
大白话解释：
损失函数结果 = 确类别概率的对数的最小化…
细节：
CrossEntropyLoss = Softmax () + 损失计算，后续如果用这个损失函数，则：输出层就不用额外调用 softmax () 激活函数了。
'''
import torch
import torch.nn as nn
def demo1():
    #如果用CrossEntropyLoss，其会把真实值转为独热编码
    y_true = torch.tensor([1,2])#独热编码[0,1,0]和[0,0,1]
    #手动设置预测值
    y_pred = torch.tensor([[0.1,0.8,0.1],[0.2,0.4,0.7]])
    #创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #计算损失值 (预测值,真实值)
    loss = criterion(y_pred,y_true)
    print(f'损失值：{loss}')
if __name__=='__main__':
    demo1()