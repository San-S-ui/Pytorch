"""
案例:
    演示学习率衰减策略。

学习率衰减策略介绍:
    目的:
        较之于AdaGrad, RMSProp, Adam方式, 我们可以通过 等间隔, 指定间隔, 指数等方式, 来手动控制学习率的调整。

    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减


等间隔学习率衰减:
    step_size: 间隔的轮数, 即: 多少轮调整一次学习率.
    gamma:     学习率衰减系数, 即: lr新 = lr旧 * gamma

指定间隔学习率衰减:
    milestones = [50, 125, 160]     里边定义的是要调整学习率的 轮数.
    gamma:     学习率衰减系数, 即: lr新 = lr旧 * gamma

指数间隔学习率衰减:
    前期学习率衰减快, 中期慢, 后期更慢, 更符合梯度下降规律.
    公式:
        lr新 = lr旧 * gamma ** epoch

总结:
    等间隔学习率衰减:
        优点:
            直观, 易于调试, 适用于 大批量数据.
        缺点:
            学习率变化较大, 可能跳过最优解.
        应用场景:
            大型数据集, 较为简单的任务.

    指定间学习率衰减:
        优点:
            易于调试, 稳定训练过程.
        缺点:
            在某些情况下可能衰减过快, 导致优化提前停滞.
        应用场景:
            对训练平稳性要求较高的任务.
   指数学习率衰减:
        优点:
            平滑, 且考虑历史更新, 收敛稳定性较强.
        缺点:
            超参调节较为复杂, 可能需要更多的资源.
        应用场景:
            高精度训练, 避免过快收敛.
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

#等间隔学习率
def demo1():
    #定义初始训练轮数，学习率,每轮训练的批次数
    epochs=200
    lr=0.1
    iteration=10
    #定义数据集
    y_true = torch.tensor([2.0],dtype=torch.float)
    x = torch.tensor([1.0],dtype=torch.float)
    w = torch.tensor([1.0],dtype=torch.float,requires_grad=True)
#创建优化器
    optimizer=optim.SGD([w],lr,0.9)


#创建等间隔学习率衰减对象
#     scheduler = optim.lr_scheduler.StepLR(optimizer,50,0.5)



#创建指定间隔学习率衰减对象
    # mile = [50,125,160]
    # scheduler=optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=mile,gamma=0.5)



#按指数学习率衰减
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.95)


#创建列表，记录训练的轮数和学习率
    epoch_list=[]
    lr_list=[]
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
#循环遍历，每轮每批进行训练
        for batch in range(iteration):
            #先计算预测值，然后基于损失函数计算损失
            y_pred = w*x
            #设置损失函数
            loss = (y_pred-y_true)**2
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.sum().backward()
            #梯度更新
            optimizer.step()
        #每轮走完后更新学习率
        scheduler.step()
    print(f"lr_list:{lr_list}")
    #可视化
    plt.plot(epoch_list,lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.show()
if __name__=='__main__':
    demo1()