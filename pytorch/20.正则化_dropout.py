#增强模型在新样本的泛化能力的策略叫做正则化
'''
正则化的作用:
    缓解模型的过拟合情况.

正则化的方式:
    L1正则化: 权重可以变为0, 相当于: 降维.
    L2正则化: 权重可以无限接近0
    DropOut: 随机失活, 每批次样本训练时, 随机让一部分神经元死亡, 防止一些特征对结果的影响较大(防止过拟合)
    BN(批量归一化): ...

对于模型参数较多，但是数据量不足的情况容易过拟合
dropout：一批次一批次的训练，每批次死亡的神经元不确定，通过组合数据，增加数据量
p->随机失活概率
存活的神经元缩放比例为1/（1-p）
测试阶段随机失活不起作用
'''
import torch
import torch.nn as nn
def demo1():
    #创建隐藏层
    t1 = torch.randint(0,10,size=(1,4)).float()
    # print(f"t1:{t1}")
    #下一层加权求和激活函数计算
    #线性层
    linear = nn.Linear(4,4)
    #加权求和
    l1=linear(t1)
    #激活函数
    output = torch.relu(l1)#relu公式：max(0,x)
    print(f'l1:{l1}')
    print(f'output:{output}')
    #进行随机失活->只有训练阶段有用
    dropout = nn.Dropout(p=0.5)#每个神经元有0.4的概率失活
    d1 = dropout(output)
    print(f'dropout:{d1}')#未被失活的缩放1/（1-p） p=0.5即扩大二倍
if __name__=='__main__':
    demo1()