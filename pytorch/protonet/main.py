"""
ProtoNet 训练 + 评估主脚本

整体流程：
  1. 准备 Omniglot 数据集（自动下载）
  2. 每个 epoch: 采样 N 个 episode 做训练
  3. 每个 epoch 后: 在验证集上评估准确率
  4. 最终: 在测试集上报告最终结果

你可以调整下面 CONFIG 字典里的参数来实验不同设定（1-shot vs 5-shot 等）
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
 
import torch.optim as optim
from tqdm import tqdm
import time

from model import Conv64, compute_prototypes, prototypical_loss
from data import OmniglotWrapper, EpisodeSampler

# ============================================================
# 配置 —— 改这里来切换实验设定
# ============================================================
CONFIG = {
    # 实验设定
    'n_way':    5,       # 每个 episode 的类别数
    'k_shot':   5,       # 每类支持样本数（1 或 5 最常用）
    'n_query':  15,      # 每类查询样本数（训练时可以用少一些）
    
    # 训练参数
    'episodes_per_epoch': 100,   # 每个 epoch 采样多少个 episode
    'epochs':             10,    # 总训练轮数
    'lr':                 0.001, # 学习率
    'weight_decay':       1e-4,  # 权重衰减
    
    # 数据
    'data_root': os.path.join(_SCRIPT_DIR, 'data'),
    
    # 设备
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ============================================================
# 训练一个 epoch
# ============================================================
def train_epoch(model, sampler, optimizer, config):
    """跑一个 epoch 的 episode 训练"""
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    device     = config['device']
    
    for _ in tqdm(range(config['episodes_per_epoch']), 
                  desc='Train', leave=False):
        # 1) 采样一个 episode
        episode = sampler.sample_episode()
        
        support = episode['support'].to(device)          # [N*K, 1, 28, 28]
        support_labels = episode['support_labels'].to(device)  # [N*K]
        query   = episode['query'].to(device)            # [N*Q, 1, 28, 28]
        query_labels = episode['query_labels'].to(device)      # [N*Q]
        
        # 2) 嵌入所有样本
        support_emb = model(support)   # [N*K, 64]
        query_emb   = model(query)     # [N*Q, 64]
        
        # 3) 计算原型
        prototypes = compute_prototypes(support_emb, support_labels, config['n_way'])
        
        # 4) 计算损失和准确率
        loss, acc = prototypical_loss(prototypes, query_emb, query_labels)
        
        # 5) 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc  += acc.item()
    
    avg_loss = total_loss / config['episodes_per_epoch']
    avg_acc  = total_acc  / config['episodes_per_epoch']
    return avg_loss, avg_acc


# ============================================================
# 评估（验证 / 测试）
# ============================================================
def evaluate(model, sampler, config, num_episodes=200):
    """在测试集上评估，返回平均准确率"""
    model.eval()
    total_acc = 0.0
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc='Eval', leave=False):
            episode = sampler.sample_episode()
            
            support = episode['support'].to(config['device'])
            support_labels = episode['support_labels'].to(config['device'])
            query   = episode['query'].to(config['device'])
            query_labels = episode['query_labels'].to(config['device'])
            
            support_emb = model(support)
            query_emb   = model(query)
            prototypes  = compute_prototypes(support_emb, support_labels, config['n_way'])
            
            _, acc = prototypical_loss(prototypes, query_emb, query_labels)
            total_acc += acc.item()
    
    return total_acc / num_episodes


# ============================================================
# 主入口
# ============================================================
def main():
    config = CONFIG
    print(f"\n设备: {config['device']}")
    print(f"设定: {config['n_way']}-way {config['k_shot']}-shot")
    print(f"训练: {config['epochs']} epochs, 每 epoch {config['episodes_per_epoch']} episodes")
    print("-" * 50)
    
    # 1) 准备数据
    print("正在准备 Omniglot 数据集（首次运行会自动下载）...")
    train_dataset = OmniglotWrapper(root=config['data_root'], background=True,  train=True)
    test_dataset  = OmniglotWrapper(root=config['data_root'], background=False, train=False)
    print(f"  训练集: {len(train_dataset)} 张图, {len(train_dataset.classes)} 个字符")
    print(f"  测试集: {len(test_dataset)} 张图, {len(test_dataset.classes)} 个字符")
    
    train_sampler = EpisodeSampler(train_dataset, config['n_way'], config['k_shot'], config['n_query'])
    test_sampler  = EpisodeSampler(test_dataset,  config['n_way'], config['k_shot'], config['n_query'])
    
    # 2) 初始化模型和优化器
    model = Conv64(in_channels=1, hidden_dim=64).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # 3) 训练循环
    best_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_sampler, optimizer, config)
        test_acc = evaluate(model, test_sampler, config, num_episodes=50)
        
        elapsed = time.time() - start
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | "
              f"Test Acc: {test_acc:.2%} | "
              f"Best: {best_acc:.2%} | "
              f"{elapsed:.1f}s")
    
    # 4) 最终测试
    print("\n" + "=" * 50)
    print("训练完成，在测试集上做最终评估...")
    final_acc = evaluate(model, test_sampler, config, num_episodes=500)
    print(f"最终 {config['n_way']}-way {config['k_shot']}-shot 准确率: {final_acc:.2%}")
    print("=" * 50)


if __name__ == '__main__':
    main()
