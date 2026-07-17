'''
张量的基本创建方式
ANN,CNN,RNN底层都是张量
存储同一类型元素的容器，且元素值必须是数值

'''
import torch
import numpy as np
def tensor_create():
    #tensor()指定数据创建
    #Tensor()指定维度创建
    #默认类型float32
    # 创建标量
    t1 = torch.tensor(10)  
    print(f't1: {t1}, type: {type(t1)}')
    print('-------------------')

    #二维列表->张量
    t2 = torch.tensor([[1, 2], [3, 4]])
    print(f't2: {t2}, type: {type(t2)}')
    print('-------------------')
    
    #numpy数组->张量
    data =np.random.randint(0,10,size=(2,3))
    t3 = torch.tensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-------------------')
   
    #torch.Tensor 可以支持指定维度的张量 tensor()方法不可以
    t4 = torch.Tensor(2,3)  #创建一个2行3列的张量，元素值为随机数
    print(f't4: {t4}, type: {type(t4)}')
    print('-------------------')


#指定数据类型创建
def tensor_create2():
    t1 = torch.IntTensor(10)  
    print(f't1: {t1}, type: {type(t1)}')
    print('-------------------')
    #二维列表->张量
    t2 = torch.DoubleTensor([[1, 2], [3, 4]])
    print(f't2: {t2}, type: {type(t2)}')
    print('-------------------')
    #类型不匹配，会自动转化
    data =np.random.randint(0,10,size=(2,3))
    t3 = torch.FloatTensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-------------------')
#指定值张量
'''
torch.ones 和 torch.ones_like 创建全 1 张量
torch.zeros 和 torch.zeros_like 创建全 0 张量
torch.full 和 torch.full_like 创建全为指定值张量
'''
def create_tensor_with_value():
    t1 = torch.ones(2,3)  #全1张量
    print(f't1: {t1}, type: {type(t1)}')
    print('-------------------')
    t2 = torch. tensor([[1, 2], [3, 4],[5, 6]])
    t3 = torch.ones_like(t2)  #参考参数格式创建全一张量
    print(f't3: {t3}, type: {type(t3)}')
    t4 = torch.zeros(2,3)  #全0张量
    print(f't4: {t4}, type: {type(t4)}')
    print('-------------------')
    t5 = torch.tensor([[1, 2], [3, 4],[5, 6]])
    t6 = torch.zeros_like(t5)  #参考参数格式创建全0张量
    print(f't6: {t6}, type: {type(t6)}')
    print('-------------------')
    #创建全为指定值张量
    t7 = torch.full(size=(2,3), fill_value=18)  #全为9张量
    print(f't7: {t7}, type: {type(t7)}')
    #创建全为指定值张量
    t8 = torch.full_like(t7, fill_value=255)  #像素越接近0越黑越接近255越白
    print(f't8: {t8}, type: {type(t8)}')

#创建线性和随机张量
def create_tensor3():
    t1 = torch.arange(0, 10, 2)  #创建一个0-10的张量，步长为2
    print(f't1: {t1}, type: {type(t1)}')
    print('-------------------')
    t2 = torch.linspace(0, 1, steps=5)  #创建一个0-1的等差张量，元素个数为5
    print(f't2: {t2}, type: {type(t2)}')
    print('-------------------')
#设置随机种子
    torch.manual_seed(42)  #设置随机种子，保证每次生成的随机数相同
    t3 = torch.rand(2, 3)  #创建一个2行3列的随机张量，元素值在[0,1)之间
    print(f't3: {t3}, type: {type(t3)}')
    print('-------------------')
    t4 = torch.randn(2, 3)  #创建一个2行3列的随机张量，元素值服从标准正态分布
    print(f't4: {t4}, type: {type(t4)}')
    print('-------------------')
    t5 = torch.randint(0, 10, size=(2, 3))  #创建一个2行3列的随机整数张量，元素值在[0,10)之间
    print(f't5: {t5}, type: {type(t5)}')

#直接创建指定类型的张量.
t1 = torch.tensor( [1, 2, 3, 4, 5], dtype=torch.float)  # 默认是：float32
print(f't1: {t1}, (元素)类型: {t1.dtype}, (张量)类型: {type(t1)}')  # float32

# 创建好张量后 → 做类型转换.
# type()函数 推荐
t2 = t1.type(torch.int16)
print(f't2: {t2}, (元素)类型: {t2.dtype}, (张量)类型: {type(t2)}')

# half() float 16/double()  float64/float() float32/short()  int 16/int()   int 32/long()  int64 整数默认

if __name__ == '__main__':
    tensor_create() 
    tensor_create2()
    create_tensor3()