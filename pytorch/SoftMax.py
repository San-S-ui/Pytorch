# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
SoftMax多用于输出层，多分类，结果以概率形式展示

'''
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 1. 定义张量，记录：分类数据．
# scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
scores = torch.tensor([[0.2, 0.35, 0.1, 0.46],[0.1,9.1,0.21,0.81]])

# 2. dim = 0，按行计算
probabilities = torch.softmax(scores, dim=0)
print(probabilities)