'''
使用 PyTorch 的 nn.MSELoss() 代替手动平方损失函数
使用 PyTorch 的 data.DataLoader 代替手动数据加载器
使用 PyTorch 的 optim.SGD 代替手动梯度下降优化器
使用 PyTorch 的 nn.Linear 代替手动写 x@w+b 假设函数
numpy数组 → Tensor张量 → TensorDataset数据集 → DataLoader数据加载器
'''
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
# 封装张量特征+标签，生成标准Dataset，给DataLoader提供数据
from torch.utils.data import DataLoader     # 数据加载器
# 自动分批、打乱、循环读取训练数据
from torch import nn                        # nn模块中有平方损失函数和假设函数
# nn.Linear 线性层(替代手写x@w+b)、nn.MSELoss 均方误差损失
from torch import optim                     # optim模块中有优化器函数
from sklearn.datasets import make_regression # 创建线性回归模型数据集
# 生成带噪声的人工回归数据集，快速做训练测试数据
import matplotlib.pyplot as plt             # 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False   
#模拟数据集
def create_datasets():
    # 1. 创建数据集对象
    x, y, coef = make_regression(
        n_samples=100,    # 100条样本
        n_features=1,     # 1个特征
        noise=10,         # 噪声，噪声越大，样本点越散
        coef=True,        # 返回真实权重系数
        bias=14.5,        # 线性偏置b
        random_state=3    # 固定随机种子，数据可复现
    )
    # 2. numpy数组转为浮点张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # 3. 返回特征、标签、真实权重
    return x, y, coef

#模型训练
def train(x, y, coef):
    #1.创建数据集对象 张量->数据集对象->数据加载器
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)#baych_size=16 每次取16条数据，shuffle=True 打乱数据
    #2 创建线性回归模型
    model = nn.Linear(in_features=1, out_features=1)#特征维度一个x对应一个y
    #3.创建损失函数对象
    criterion = nn.MSELoss()#均方误差损失函数
    #4.创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr=0.01)#随机梯度下降 
    #开始训练
    epochs,sample_total,total_loss,loss_list = 100, 0, 0, []
    for epoch in range(epochs):
        for train_x,train_y in dataloader:
            #1.梯度清零
            optimizer.zero_grad()
            #2.正向传播
            y_pred = model(train_x)
            #3.计算损失
            loss = criterion(y_pred, train_y.reshape(-1, 1))#train_y.reshape(-1, 1) 所有行 reshape成1列
            #4.反向传播
            loss.backward()
            #5.更新参数
            optimizer.step()
            total_loss += loss.item()
            sample_total +=1
        loss_list.append(total_loss/sample_total)
        #打印结果
        print(f"轮数：{epoch+1}平均损失值：{total_loss/sample_total}")
    print(f"{epochs}轮的平均损失是{loss_list}")
    print(f"模型权重：{model.weight}，偏置{model.bias}")
             
            
#程序入口
if __name__ == '__main__':
     x, y, coef = create_datasets()
    #  print(f'x形状: {x.shape}, y形状: {y.shape}, 真实权重coef: {coef}')
     train(x,y,coef)