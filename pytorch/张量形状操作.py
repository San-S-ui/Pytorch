#reshape unsqueeze transpose permute view contiguous is_contiguous 

import torch
import random
torch.manual_seed(42)
# def demo():
#     t1 = torch.randint(1,10,size=(2,3))
#     print(f't1:{t1},row:{t1.shape[0]},colums:{t1.shape[1]}')
#     #通过reshape 转换t1的类型 2*3=6所以元素必须是6  不改变内容
#     t2 = t1.reshape(3,2)
#     t3 = t1.reshape(1,6)
#     print(f't2:{t2},row:{t2.shape[0]},colums:{t2.shape[1]}')
#     print(f't3:{t3},row:{t3.shape[0]},colums:{t3.shape[1]}')
    #通过unsqueeze 在指定的维度上加一个维度
    #squeeze 删除所有为1的维度 
    # transpose() permute()  交换维度 不改变源数据
    #view 只可以修改连续的 用transpose() permute() 处理后的张量不可以用view
'''view() 只能修改连续的张量的形状，连续张量 =内存中存储顺序 和 在张量中显示的顺序相同。contiguous() 把不连续的张量 → 连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序。is_contiguous() 判断张量是否是连续的。
'''
def demo1():
    t1 = torch.randint(1,10,size=(2,3,1))
    print(f't1:{t1},{t1.shape}')
    t2 = t1.unsqueeze(0) #0维加
    t3 = t1.unsqueeze(1)
    t4 = t1.squeeze()
    print(f't2:{t2},{t2.shape}')
    print(f't3:{t2},{t3.shape}')
    print(f't4:{t4},{t4.shape}')    
    print('-'*30)
    a1 = torch.randint(1,10,size=(2,3))
    a3 = a1.transpose(0,1)
    a4 = a1.permute(1,0)
    print(f'a1:{a1},{a1.shape}')
    print(f'a3:{a3},{a3.shape}')  
    print(f'a4:{a4},{a4.shape}') 
    a2 = a1.view(3,2)
    print(f'a2:{a2},{a2.shape}')
    print(a2.is_contiguous())
    a5 = a3.contiguous().view(2,3)#a3不连续先用contiguoue转为连续的再用view转
    print(f'a5:{a5},{a5.shape}')
if __name__=='__main__':
    # demo()
    demo1()