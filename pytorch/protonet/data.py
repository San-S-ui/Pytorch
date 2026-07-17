"""
数据加载 —— Omniglot + Episode 采样器

小样本学习不按传统 batch 训练，而是以"episode（任务）"为单位：
  每个 episode = 从 N 个类别中，每类取 K 个支持样本 + Q 个查询样本
  这就是 N-way K-shot 设定
"""
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class RandomRotation90:
    """随机旋转 0/90/180/270 度 —— Omniglot 的标准增广"""
    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, dims=[-2, -1])


class OmniglotWrapper(Dataset):
    """Omniglot 封装，按类别分组，便于 episode 采样

    数据集说明：
      - Omniglot 包含 1623 个手写字符，每个字符 20 张图
      - background (训练) = 1200 个字符，evaluation (测试) = 423 个字符
      - 原始图片 105x84，统一缩放到 28x28
      - 训练时做 0/90/180/270 旋转增广
    """
    def __init__(self, root='./data', background=True, train=True):
        base_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])

        self.dataset = datasets.Omniglot(
            root=root, background=background, download=True,
            transform=base_transform,
        )

        # 按类别收集样本索引
        self.class_to_indices = {}
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.train = train
        self.rotation = RandomRotation90()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.train:
            img = self.rotation(img)
        return img, label


class EpisodeSampler:
    """小样本 Episode 采样器

    每个 episode 的形状（5-way 5-shot 为例）：
      support:  [25, 1, 28, 28]   5 类 x 5 张
      query:    [75, 1, 28, 28]   5 类 x 15 张

    使用方式：
      sampler = EpisodeSampler(dataset, n_way=5, k_shot=5, n_query=15)
      episode = next(sampler)
    """
    def __init__(self, dataset, n_way, k_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def sample_episode(self):
        """生成一个完整的 episode 并返回"""
        episode_classes = random.sample(self.dataset.classes, self.n_way)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for i, cls in enumerate(episode_classes):
            indices = self.dataset.class_to_indices[cls].copy()
            random.shuffle(indices)

            sup_idx = indices[:self.k_shot]
            qry_idx = indices[self.k_shot:self.k_shot + self.n_query]

            for idx in sup_idx:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(i)

            for idx in qry_idx:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(i)

        return {
            'support':        torch.stack(support_images),      # [N*K, 1, 28, 28]
            'support_labels': torch.tensor(support_labels),     # [N*K]
            'query':          torch.stack(query_images),        # [N*Q, 1, 28, 28]
            'query_labels':   torch.tensor(query_labels),      # [N*Q]
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample_episode()
