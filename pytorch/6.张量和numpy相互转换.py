'''
张量和numpy相互转换
.detach().numpy()方法：张量转numpy 不共享内存
detach()复制张量，返回一个新的张量，和原来的张量不共享内存

.numpy().copy()方法：张量转numpy 不共享内存

from_numpy()方法：numpy转张量 共享内存
torch.tensor()方法：numpy转张量 不共享内存
'''
import torch
import numpy as np
# def tensor_numpy():
#     # 张量转numpy
#     t1 =torch.tensor([1,2,3,4,5])
#     n1 = t1.numpy()
#     print(t1,type(t1))
#     print(n1,type(n1))
#     print('-'*50)
#     #测试共享内存
#     n1[0]=100
#     print(t1,type(t1))
#     print(n1,type(n1))
def tensor_numpy1():
    # numpy转张量
    n1 =np.array([5,4,3,2,1])
    t1= torch.from_numpy(n1)
    print(t1,type(t1))
    print(n1,type(n1))
    print('-'*50)
    #测试共享内存
    t3 = torch.tensor(n1) 
    n1[0]=100
    print(t1,type(t1))
    print(n1,type(n1))
    print(t3,type(t3))
def demo():
    #从标量张量中提取值
    t1 = torch.tensor(100)
    a = t1.item()
    print(t1,type(t1))
    print(a,type(a))
if __name__=='__main__':
    # tensor_numpy()
    # tensor_numpy1()
    demo()


    
