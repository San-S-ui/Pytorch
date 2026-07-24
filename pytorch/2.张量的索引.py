import torch
torch.manual_seed(42)
t1 = torch.randint(1,10,(5,5))
print(t1)
#简单行列
print(t1[1])
print(t1[1,:2]) #第一行前两列
print(t1[:,2])#所有行第二列
print('-'*30)
#列表索引
print(t1[[2,1],[3,4]]) #返回[2，3]和[1，4]的内容
#范围索引
print(t1[:3,:2]) #前3行前2列
print(t1[1::2,::2])#奇数行,偶数列
#布尔索引
print(t1[torch.tensor([True,False,False,False,True]),:])#第0行和最后一行
print(t1[t1[:,2]>5])#第2列大于5的行数据
print(t1[:,t1[2]>5])#第2行大于5的列数据
print(t1[2,t1[2]>5])
print('-'*30)
#多维索引
#创建一个三维张量
"""
[  0轴
    [1轴
        [2轴
        5,
        3,
        1],
        [2, 2, 4]
    ],
    [
        [4, 8, 8],
        [9, 8, 3]
    ]
]
"""
t2 = torch.randint(1,10,(2,2,3))
print(f"t2:{t2}")
print(t2[0,:,:])
print(t2[:,0,:])
print(t2[:,:,0])
