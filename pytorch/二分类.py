"""
案例:
    演示二分类任务的损失函数.

二分类任务的损失函数(BCELoss):
    公式:
        Loss = -ylog(预测值) - (1 - y)log(1 - 预测值)
    细节:
        因为公式中没有包含Sigmoid激活函数, 所以使用BCELoss的时候, 还需要手动指定 Sigmoid.
"""

import torch
import torch.nn as nn
def demo1():
    #真实值不会转为独热编码了所以按之前那样写不行了
    y_true = torch.tensor([1,0,1],dtype=torch.float)
    #手动设置预测值
    y_pred = torch.tensor([0.53,0.1,0.639])#是1的概率是0.53，是0的概率是0.1，是1的概率是0.639
    #创建二分类交叉熵损失函数
    m=nn.Sigmoid()
    criterion = nn.BCELoss()
    #计算损失值 (预测值,真实值)
    #y_pred 已经是 0~1 概率，不需要再套 Sigmoid
    # loss = criterion(m(y_pred),y_true).detach().numpy()
    
    loss = criterion(y_pred,y_true)
    print(f'损失值：{loss}')
if __name__=='__main__':
    demo1()