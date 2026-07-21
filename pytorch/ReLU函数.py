# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
 ReLU多用于隐藏层，深层层神经网络，小于0映射为0（神经元死亡）,梯度小于0也映射为0,默认只考虑正样本


'''
import torch
import matplotlib.pyplot as plt
# 中文、负号显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 1. 创建画布和坐标轴，1行2列．
fig, axes = plt.subplots( nrows=1,  ncols=2)

# 2. 生成 -20 ~ 20 之间的 1000个数据点．
x = torch.linspace(-20, 20, 1000)
# print(f'x: {x}')

# 3. 计算上述1000个点， relu激活函数处理后的值．
y = torch.relu(x)
# print(f'y: {y}')

# 4. 在第1个子图中绘制 relu激活函数的图像．
axes[0].plot(x, y)
axes[0].set_title(' relu激活函数图像')
axes[0].grid()

# 5．在第2个图上，绘制 relu激活函数的导数图像．
# 5.1 重新生成 -20 ~ 20 之间的 1000个数据点．
# 参1：起始值，参2：结束值，参3：元素的个数，参4：是否需要求导．
x = torch.linspace(-20, 20, 1000, requires_grad=True)

# 5.2 具体的计算上述1000个点，simgoid激活函数导数后的值．
torch.relu(x).sum().backward()

# 5.3 绘制图像．
axes[1].plot(x.detach(), x.grad)
axes[1].set_title('relu激活函数导数图像')
axes[1].grid()
plt.show()
plt.close()
