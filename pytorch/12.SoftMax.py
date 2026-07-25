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
# 3行2列
# [[1, 2],
#  [3, 4],
#  [5, 6]]
# dim=0：沿着行运算，按每一列做 softmax
# 对每一列单独算概率，同一列总和 = 1
# 列 1：1,3,5 → softmax 后三个数相加 = 1
# 列 2：2,4,6 → softmax 后三个数相加 = 1
# ② dim=1：沿着列运算，按每一行做 softmax
# 对每一行单独算概率，同一行总和 = 1
# 第一行 1,2 → 和为 1
# 第二行 3,4 → 和为 1
# 第三行 5,6 → 和为 1