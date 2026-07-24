import torch
# #可以直接用符号，也可以用函数
# t1 = torch.tensor([1,2,3,4,5])
# t2 = torch.tensor(10)
# t3 = t1.add(t2) #不修改t1
# # t3 = t1+10  #同上
# print(t3)
# print(t1)
# t1.add_(t2) #修改t1
# # t1+=10#同上
# print(t1)
# # 减sub() 乘 mul()  除div() 保留小数/ 整除是//  neg()加负数

# #点乘 mul() 或者*  两个向量对应位置相乘
# t1 = torch.tensor([[1,2],[3,4]])
# t2 = torch.tensor([[5,6],[7,8]])
# t3 = torch.mul(t1,t2)
# print(t1*t2)

# #矩阵乘法 行乘列要符合条件shape1(n,m)shape2(m,p)
# #  @  matmul()  
# t1 = torch.tensor([[1,2],[3,4],[5,6]])
# t2 = torch.tensor([[8,7,2],[9,7,3]])
# t3 = torch.matmul(t1,t2)
# print(t1@t2)
# print(t3)

#张量运算函数
'''涉及到的 API (函数) 如下：
sum (), max (), min (), mean ()→ 都有 dim 参数，0 表示列，1 表示行
pow (), sqrt (), exp (), log (), log2 (), log10 ()　　→ 没有 dim 参数
掌握的函数：
sum (), max (), min (), mean (), pow ()
'''
# #求和
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.sum(dim=0))
# print(t1.sum(dim=1))
# print(t1.sum())
# #求最大值
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.max(dim=0)) #对比列的最大值
# print(t1.max(dim=1))
# print(t1.max())
# #求平均值
# t1 = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)
# print(t1.mean(dim=0)) #对比列的平均值 只可以是float
# print(t1.mean(dim=1))
# print(t1.mean())
# #求每个数的平方
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.pow(2)) 
# print(t1**2)
#开方
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.sqrt()) 
# #n次幂
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.exp()) #1*1 2*2 3*3 ....
#对数
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.log()) #以1为底
# print(t1.log2()) #以2为底
# print(t1.log10()) #以10为底