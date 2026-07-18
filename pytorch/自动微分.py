#自动微分模块，对损失函数求导，结合反向传播，更新权重参数w(权重) b(偏执) 多元线性公式y=wx+b w新=w旧-学习率*梯度,大部分不考虑偏执，直接把偏执矩阵设为全0矩阵 梯度=损失函数的导数
import torch
# #pyorch只支持标量张量对向量张量求导
# #forward()正向传播 backward()自动计算梯度反向传播
# # 1.只有标量才可以backward(求导)底层大多数都是浮点型 requires_grad是自动微分   记录初始权重 w旧
# w =torch.tensor(10,requires_grad=True,dtype=torch.float)
# #2.定义loss函数，表示损失函数
# loss = 2 * w**2
# #3.计算梯度 计算完毕后记录到w.grad()
# loss.backward()
# #4.带入权重更新公式 w新=w旧-学习率*梯度
# w.data = w.data-0.01*w.grad
# #打印最终结果
# print(f"w更新后的值{w}")

"""
自动微分模块，循环实现计算梯度，更新参数
"""
#设置初始w旧
w = torch.tensor(10,requires_grad=True,dtype=torch.float)
#定义损失函数
loss = w**2 + 20
print(f'权重初始值：{w},（0.01*w.grad）:无,loss:{loss}')
#正向计算
for i in range(1,101):
    loss = w**2+20 #梯度2w
    
#梯度清零 要不然梯度会累加
    if w.grad !=None:
        w.grad.zero_()
#反向传播 产生w.grad
    loss.sum().backward()
#梯度更新
    w.data = w.data-0.01*w.grad

    print(f'第{i}次，权重初始值：{w},（0.01*w.grad）:{0.01*w.grad:.5f},loss:{loss:.5f}')
#打印权重
print(f'第{i}次，权重：{w},梯度:{w.grad},loss:{loss}')