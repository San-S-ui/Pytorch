"""
案例:
    RNN案例, 基于杰伦歌词来训练模型, 用给定的起始词, 结合长度, 来进行 AI歌词生成.

实现步骤:
    1. 获取数据, 进行分词, 获取词表.
    2. 数据预处理, 构建数据集.
    3. 搭建RNN神经网络.
    4. 训练模型.
    5. 模型预测.
"""

# 导包
import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

def build_vocab():
    #定义列表，记录去重后所有的词, 每行文本分词结果.
    unique_words,all_words=[],[]
    for line in open('D:/pytorch/pytorch/data/RNNdata/jaychou_lyrics.txt','r',encoding='UTF-8'):
        words = jieba.lcut(line)
        all_words.append(words)
        # print(all_words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    #去重后的词数量
    word_count = len(unique_words)
    # print(unique_words)
    # print(word_count)#5703
    #构建词表 i是索引
    word_index = {word:i  for i,word in enumerate(unique_words)}
    # print(f"word_index:{word_index}")
    #把所有歌词文本用索引表示
    cor_index=[]
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_index[word])
        #在每行词之间, 添加空格隔开.
        tmp.append(word_index[' '])
        cor_index.extend(tmp)
    return unique_words, word_index, word_count, cor_index


#数据预处理，构建数据集
# 定义数据集类, 继承 torch.utils.data.Dataset
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self,cor_index,num_chars):
        super().__init__()
        #文档中词的索引
        self.cor_index = cor_index
        #每个句子中词的个数
        self.num_chars = num_chars
        #文档中词的个数
        self.word_count = len(self.cor_index)
        #句子数量
        self.number = self.word_count//self.num_chars
    #当使用 len(obj)时, 自动调用此方法.
    def __len__(self):
        #返回句子数量
        return self.number
     #当使用 obj[index]时, 自动调用此方法.
    def __getitem__(self, index):
        #防止索引越界
        start = min(max(index,0),self.word_count-self.num_chars-1)
        end = start+self.num_chars
        #获取输入值x
        x = self.cor_index[start:end]
        #获取输出值y
        y = self.cor_index[start+1:end+1]
        return torch.tensor(x),torch.tensor(y)

# 3. 搭建RNN神经网络.
class TextGenerator(nn.Module):
    # 初始化方法
    def __init__(self, word_count):      # word_count: 去重的词的数量(5703)
        #初始化父类的成员.
        super().__init__()
        #初始化词嵌入层
        self.ebd = nn.Embedding(word_count,128)
        #循环网络层：词向量维度，隐藏层维度：256，网络层：1
        self.rnn = nn.RNN(128,256,1)
        #输出层 特征向量维度(和隐藏向量维度一致), 词表中词的个数.
        self.out = nn.Linear(256,word_count)
    #前向传播
    def forward(self,inputs,hidden):
        #初始化 词嵌入层处理.
        # embd格式: (batch句子的数量, 句子的长度, 词向量维度)
        embd = self.ebd(inputs)

        # rnn处理
        # rnn格式: (句子的长度, batch句子的数量, 隐藏层维度)
        output, hidden = self.rnn(embd.transpose(0, 1), hidden)

        #全连接层 第二维固定是 hidden_dim，因此第一维自动算
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 变形前：(seq_len, batch_size, hidden_dim) 变形后：(seq_len * batch_size, hidden_dim)

        #返回结果, 预测结果, 隐藏层.
        return output, hidden
    #隐藏层的初始化方法.
    def init_hidden(self, bs):      # batch_size
    # 隐藏层初始化: [网络层数, batch, 隐藏层向量维度]
         return torch.zeros(1, bs, 256)

#训练模型.
def train():
     # 1. 构建词典.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 2. 获取数据集.
    lyrics = LyricsDataset(corpus_idx, 32)
    # 3. 初始化(神经网络)模型
    model = TextGenerator(unique_word_count)        # 预测5703个词, 每个词的概率.
    # 4. 创建数据加载器对象.
    # 参1: 数据集对象.  参2: 批次大小(每批5个句子, 每个句子32个词)  参3: 是否打乱数据.
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 6. 定义优化器.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 7. 模型训练.
    # 7.1 定义变量, 记录训练的轮数.
    epochs = 50
    # 7.2 具体的每轮训练动作.
    for epoch in range(epochs):     # epoch: 0, 1, 2, 3...9, 分别表示: 第1轮, 第2轮, ... 第10轮.
        # 7.3 定义变量记录: 本轮开始训练时间, 迭代(批次)次数, 训练总损失.
        start, iter_num, total_loss = time.time(), 0, 0.0
        # 7.4 具体的 本轮 各批次 训练动作.
        # 遍历数据集, 后台会调用 LyricsDataset#__getitem__()方法, 获取到每个样本的数据和标签,
        for x, y in lyrics_dataloader:
            # 7.5 获取隐藏层初始值.
            hidden = model.init_hidden(5)
            # 7.6 模型计算.
            output, hidden = model(x, hidden)
            # 7.7 计算损失.
            # y的形状: (batch 批次数, seq_len 句子长度, 词向量维度) -> 转成一维向量 -> 每个词的下标索引.
            # output形状为: (seq_len, batch, 词向量维度)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))
            loss = criterion(output, y)
            # 7.8 梯度清零 + 反向传播 + 更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 7.9 累计损失 和 迭代次数.
            total_loss += loss.item()
            iter_num += 1

        # 7.10 走到这里, 说明 本轮训练结束, 打印本轮的训练信息.
        print(f'epoch: {epoch + 1}, time: {time.time() - start:.2f}s, loss: {total_loss / iter_num:.4f}')
            
    # 8. 走到这里, 说明多轮训练结束(模型训练结束), 保存即可.
    torch.save(model.state_dict(), 'D:/pytorch/pytorch/model/text_generator.pth')
# 5. 模型预测.
def evaluate(start_word, sentence_length):
    # 1. 构建词典.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    #获取模型
    model = TextGenerator(unique_word_count)
    # 3. 加载模型参数.
    model.load_state_dict(torch.load('D:/pytorch/pytorch/model/text_generator.pth'))
    # 4. 获取隐藏层初始值.
    hidden = model.init_hidden(1)
    # 5. 将输入的 开始词 转换成 索引.
    word_idx = word_to_index[start_word]
    # 6. 定义列表, 存放: 产生的词的索引.
    generate_sentence = [word_idx]  # 开始词的索引, 是列表的: 第1个值.
    # 7. 遍历句子长度, 获取到每一个词.
    for i in range(sentence_length):
        # 7.1 模型预测.
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 7.2 获取预测结果.   argmax() 从所有结果(5703个词的概率)中, 找最大值对应的索引.
        word_idx = torch.argmax(output)
        # 7.3 把预测结果添加到列表中.
        generate_sentence.append(word_idx)

    # 8. 将索引转成词, 并打印.
    for idx in generate_sentence:
        print(unique_words[idx], end='')

if __name__=='__main__':
    # 获取数据, 进行分词, 获取词表.
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print(f'词的数量: {word_count}')         # 去重后, 5703个词
    
    # print(f'去重后的词: {unique_words}')     # ['想要', '有', '直升机', '\n', '和', '你'..., '要大卖']
    
    # print(f'每个词的索引: {word_to_index}')  # 词表: {'想要': 0, '有': 1, '直升机': 2, '\n': 3, '和': 4, '你': 5, ... ', '要大卖': 5702}
    
    # print(f'文档中每个词对应的索引: {corpus_idx}')  # [0, 1, 2, 1 3, 40, 0, 4, 5, 6, 7, 8, 3, 40,......]
    
    # 构建数据集
    # dataset = LyricsDataset(cor_index=corpus_idx,num_chars=5 )
    # print(f'句子数量: {len(dataset)}')
   
    # 查看下 输入值 和 目标值.
    # x, y = dataset[0]
    # print(f'输入值: {x}')  
    # print(f'目标值: {y}')  

    train()

     # 测试模型.
    # evaluate('星星', 50)