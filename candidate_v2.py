"""候选 v2（实验）：迁移学习 —— 用 ImageNet 预训练的 ResNet18 提取特征，只训练最后一层线性分类器。

思路（三个概念）：
1. 特征提取：预训练 ResNet18 已经"见过"海量真实物体，能学会把图片变成一串特征数字（512 个）；
2. 冻结：我们不改预训练模型的权重（p.requires_grad = False），它只负责"把图变成特征"；
3. 分类器：用一个新的 512→4 的线性分类器，学习"特征 → 四类垃圾"。

和候选 v1（小 CNN 0.6268）用同一个 split.json、同一个测试集 2417 张比较（同条件）。
这是独立实验，不影响主流程。
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

SPLIT_FILE = Path("split.json")
IMG_SIZE = 224        # ResNet 要求的输入大小
TRAIN_LIMIT = 4000    # 和候选 v1 一样：每类最多 1000
EPOCHS = 30           # 线性分类器收敛快，多给点
BATCH = 64
LR = 0.001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# 预训练模型要求的预处理：缩放到 224 + 转张量 + ImageNet 归一化
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_tensor(path: str):
    """把一张图片变成预训练模型能吃的张量。"""
    return preprocess(Image.open(path).convert("RGB"))


def main() -> int:
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)

    labels = sorted(split["TRAIN"].keys())
    label2idx = {name: i for i, name in enumerate(labels)}
    print("4 个类别:", labels)

    # 1) 预训练 ResNet18，去掉最后的分类层，当作特征提取器
    print("加载 ImageNet 预训练 ResNet18（第一次会下载约 45MB 权重，请耐心等待）...")
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()      # 输出变成 512 维特征
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False      # 冻结：不改预训练权重

    # 2) 训练/测试数据（和候选 v1 一样的子集和划分）
    per_class = max(1, TRAIN_LIMIT // len(labels))
    train_paths, train_y = [], []
    for label in labels:
        paths = split["TRAIN"][label][:per_class]
        train_paths += paths
        train_y += [label2idx[label]] * len(paths)
    test_paths, test_y = [], []
    for label in labels:
        test_paths += split["TEST"][label]
        test_y += [label2idx[label]] * len(split["TEST"][label])
    print(f"训练 {len(train_paths)} 张 / 测试 {len(test_paths)} 张")

    # 3) 用预训练模型把每张图变成 512 维特征（只前向，一次性算完）
    def extract(paths):
        feats = []
        with torch.no_grad():
            for p in paths:
                x = load_image_tensor(p).unsqueeze(0)
                feats.append(backbone(x))
        return torch.cat(feats)

    print("提取训练特征（约几分钟）...")
    train_x = extract(train_paths)
    train_y = torch.tensor(train_y)
    print("提取测试特征...")
    test_x = extract(test_paths)
    test_y = torch.tensor(test_y)

    # 4) 训练一个简单的线性分类器（512 → 4）
    model = nn.Linear(train_x.shape[1], len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    n = len(train_x)
    for epoch in range(EPOCHS):
        order = list(range(n))
        random.shuffle(order)
        total_loss, steps = 0.0, 0
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            out = model(train_x[idx])
            loss = loss_fn(out, train_y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
        print(f"epoch {epoch + 1}/{EPOCHS}  平均 loss = {total_loss / steps:.4f}")

    # 5) 评估（和候选 v1 同一个测试集）
    model.eval()
    with torch.no_grad():
        preds = model(test_x).argmax(dim=1)
    correct = (preds == test_y).sum().item()
    total = len(test_y)
    acc = correct / total
    print(f"\n候选 v2（预训练 ResNet + 线性分类器）准确率 = {acc:.4f}  ({correct}/{total})")
    print(f"候选 v1（小 CNN）                    = 0.6268")
    print(f"差值                                 = {acc - 0.6268:+.4f}")

    # 每类准确率（沿用之前的发现，看可回收物是否改善）
    print("\n每类准确率（测试集内）：")
    for cls in labels:
        idx = label2idx[cls]
        n_cls = int((test_y == idx).sum())
        ok_cls = int(((preds == test_y) & (test_y == idx)).sum())
        print(f"  {cls}: {ok_cls}/{n_cls} = {ok_cls / n_cls:.4f}")

    # 失败案例数（和候选一对比）
    n_fail = int((preds != test_y).sum())
    print(f"失败案例数：{n_fail}")

    # 保存训练好的线性分类器，供 predict.py 现场演示（迁移学习版）
    torch.save(model.state_dict(), "model_v2.pt")
    print("已保存 model_v2.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
