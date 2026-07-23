"""
案例:
    演示 回归任务的损失函数介绍.

回归任务常用损失函数如下:

MAE:  Mean Absolute Error, 平均绝对误差.
    公式:
        误差绝对值之和 / 样本总数
    类似于L1正则化, 权重可以降维0, 数据会变得稀疏.

    弊端:
        在0点不平滑, 可能错过最小值.

MSE:  Mean Squared Error, 均方误差.
可以用来快速找出异常值，常用于正则化
如果差值过大，可能存在梯度爆炸

Smooth L1:
    将MAE和MSE结合起来了，在x=[-1,1]这个区间用MSE
    
"""


import torch
import torch.nn as nn
def demo1():
    
    y_true = torch.tensor([2,2,1],dtype=torch.float)
    #手动设置预测值
    y_pred = torch.tensor([1,1.9,0.92],requires_grad=True)#是1的概率是0.53，是0的概率是0.1，是1的概率是0.639
    #MAE
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.SmoothL1Loss()
    loss1 = criterion1(y_pred,y_true)
    loss2 = criterion2(y_pred,y_true)
    loss3 = criterion3(y_pred,y_true)
    print(f'MAE：{loss1}, MSE：{loss2},  SmoothL1{loss3}')
if __name__=='__main__':
    demo1()