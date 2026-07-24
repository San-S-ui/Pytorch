"""
案例:
演示参数初始化的 7 种方式。

参数初始化的目的:
1. 防止梯度消失 或者 梯度爆炸。
2. 提高收敛速度。
3. 打破对称性。

参数初始化的方式:
无法打破对称性的:
    全0，全1，固定值
可以打破对称性的:
    随机初始化，正态分布初始化，kaiming初始化，xavier初始化

总结:
1. 记忆 kaiming初始化，xavier初始化，全0初始化。
2. 关于初始化的选择上:
    激活函数ReLU及其系列: 优先用 kaiming
    激活函数非ReLU: 优先用 xavier
    如果是浅层网络: 可以考虑使用 随机初始化
"""
import torch
from torch import nn
#随机分布初始化
def dome1():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)#权重
    print(linear.bias.data)#偏置
#固定初始化
def dome2():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    nn.init.constant_(linear.weight,3)
    nn.init.constant_(linear.bias,3)
    print(linear.weight.data)#权重
    print(linear.bias.data)#偏置
#全零初始化
def dome3():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)#权重
    print(linear.bias.data)#偏置
#全1初始化
def dome4():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)#权重
    print(linear.bias.data)#偏置
#正态分布初始化
def dome5():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    nn.init.normal_(linear.weight)
    nn.init.normal_(linear.bias)
    print(linear.weight.data)#权重
    print(linear.bias.data)#偏置
#kaiming正态分布初始化
def dome6():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    #kaiming正态分布初始化 只可以初始化weight
    nn.init.kaiming_normal_(linear.weight)  
    # #kaiming均匀分布初始化
    # nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight.data)#权重
#xavier正态分布初始化
def dome7():
    #设置线性层
    linear =nn.Linear(5,3)#五个输入三个输出 
    #xavier正态分布初始化 只可以初始化weight
    nn.init.xavier_normal_(linear.weight)  
    # #xavier均匀分布初始化
    # nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)#权重
if __name__=='__main__':
    # dome1()
    # dome2()
    # dome3()
    # dome4()
    # dome5()
    # dome6()
    dome7()