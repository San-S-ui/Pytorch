"""
模型定义 —— 嵌入网络 + 原型损失

论文: "Prototypical Networks for Few-shot Learning" (Snell et al., NeurIPS 2017)

核心逻辑很简单：
  1. 用一个 CNN 把图片映射到 64 维特征空间
  2. 每个类别用"支持集"特征的均值作为该类的「原型」
  3. 查询样本的分类 = 找最近的原型（欧氏距离）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv64(nn.Module):
    """4 层卷积嵌入网络，输出 64 维特征向量。
    
    这是 ProtoNet 原论文为 Omniglot (28x28 灰度图) 设计的标准网络。
    
    结构变化：
      28x28 -> (Conv+BN+ReLU+Pool)x4 -> 1x1x64 -> flatten -> 64 维向量
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: 3x3 -> 1x1
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),       # [B, 64, 1, 1] -> [B, 64]
        )

    def forward(self, x):
        return self.encoder(x)


def compute_prototypes(support_embeddings, support_labels, n_way):
    """计算每个类别的原型（支持集嵌入的均值）
    
    Args:
        support_embeddings: [n_way * k_shot, d] 所有支持样本的特征
        support_labels:     [n_way * k_shot]    对应类别标签 (0..n_way-1)
        n_way:              本次 episode 的类别数
    
    Returns:
        prototypes: [n_way, d] 每个类别的原型向量
    """
    d = support_embeddings.size(-1)
    prototypes = torch.zeros(n_way, d, device=support_embeddings.device)
    for i in range(n_way):
        mask = (support_labels == i)
        prototypes[i] = support_embeddings[mask].mean(dim=0)
    return prototypes


def prototypical_loss(prototypes, query_embeddings, query_labels):
    """原型损失：查询样本到各原型的负对数概率
    
    流程：
      1. 计算每个查询嵌入到所有原型的欧氏距离  [n_query, n_way]
      2. 距离取负 -> logits（距离越小，logit 越大）
      3. softmax -> 交叉熵
    
    Returns:
        loss: 标量损失
        acc:  查询集分类准确率
    """
    dists = torch.cdist(query_embeddings, prototypes)
    logits = -dists
    loss = F.cross_entropy(logits, query_labels)
    pred = logits.argmax(dim=1)
    acc = (pred == query_labels).float().mean()
    return loss, acc
