import torch
from torch import nn
from torchsummary import summary#计算模型参数，查看模型结构

#搭建神经网络 继承nn.Module
class ModelDemo(nn.Module):
    def __init__(self):
        #初始化父类，这样才可以用父类的构造方法
        super().__init__()
        #搭建神经网络 
        #隐藏层1：3个输出3个输入
        self.linear1 = nn.Linear(3,3)
        #隐藏层2：3个输出2个输入
        self.linear2 = nn.Linear(3,2)
        #输出层
        self.output = nn.Linear(2,2)


        #参数初始化 初始化w和b
        #隐藏层一初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        #隐藏层二初始化
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


    #前向传播 输入—>隐藏->输出
    def forward(self,x):
        #加权求和+激活函数
        #隐藏层一计算 xavier_normal+sigmoid
        x= torch.sigmoid(self.linear1(x))
        #隐藏层二计算
        x= torch.relu(self.linear2(x))
        #输出层计算
        x= torch.softmax(self.output(x),dim=-1)#按行计算
        #返回预测值
        return x
#训练模型
def train():
    #生成模拟数据
    data = torch.randn(size=(5,3))
    print(f'data:{data}')
    print(f'data.shape:{data.shape}')
    print(f'data.requires_grad:{data.requires_grad}')    
    #创建模型对象
    my_model = ModelDemo()

    #通过神经网络模型训练
    output = my_model(data)#自动将requires_grad改为True
    print(f'data:{output}')
    print(f'output.shape:{output.shape}')
    print(f'output.requires_grad:{output.requires_grad}')   
    print('==========================计算模型参数=======================')
    summary(my_model,input_size=(5,3))
    print('==========================查看模型参数=======================')
    for name,param in my_model.named_parameters():
        print(f"name:{name}")
        print(f'param:{param}\n')


if __name__=='__main__':
    train()