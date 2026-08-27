"""4 类候选方法：小型 CNN，看图片像素，学区分 可回收/厨余/有害/其他。

读取 make_split.py 生成的 split.json（和基线同一份划分，同条件比较）。
训练只用训练集一部分（TRAIN_LIMIT），几分钟内能跑完。
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

# ---------- 配置 ----------
SPLIT_FILE = Path("split.json")
IMG_SIZE = 64        # 图片缩到 64x64（越小训练越快）
TRAIN_LIMIT = 4000        # 训练子集总张数（平均分给 4 类）
EPOCHS = 6
BATCH = 32
LR = 0.001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def paths_to_tensor(paths):
    """把所有图片变成张量 [张数, 3, 64, 64]，数值 0~1。"""
    tensors = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = torch.tensor(list(im.get_flattened_data()), dtype=torch.float32)
        arr = arr.t().reshape(3, IMG_SIZE, IMG_SIZE) / 255.0
        tensors.append(arr)
    return torch.stack(tensors)


class TinyCNN(nn.Module):
    """两层卷积+池化，再接全连接，输出 num_classes 个类别分数。"""
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * (IMG_SIZE // 4) * (IMG_SIZE // 4), num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def main() -> int:
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)

    # 类别按名字排序，编号固定（保证可复现）
    labels = sorted(split["TRAIN"].keys())
    label2idx = {name: i for i, name in enumerate(labels)}
    idx2label = {i: name for i, name in enumerate(labels)}
    print("4 个类别:", labels)

    # 1) 训练数据：每类最多取 per_class 张
    per_class = max(1, TRAIN_LIMIT // len(labels))
    train_paths, train_y = [], []
    for label in labels:
        paths = split["TRAIN"][label][:per_class]
        train_paths += paths
        train_y += [label2idx[label]] * len(paths)
    print(f"训练子集：共 {len(train_paths)} 张（每类最多 {per_class} 张）")
    train_x = paths_to_tensor(train_paths)
    train_y = torch.tensor(train_y)

    # 2) 测试数据：完整测试集（和基线同一份）
    test_paths, test_y = [], []
    for label in labels:
        test_paths += split["TEST"][label]
        test_y += [label2idx[label]] * len(split["TEST"][label])
    test_x = paths_to_tensor(test_paths)
    test_y = torch.tensor(test_y)
    print(f"测试集：共 {len(test_paths)} 张")

    # 3) 训练
    model = TinyCNN(len(labels))
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

    # 4) 评估（和基线同条件）
    model.eval()
    with torch.no_grad():
        preds = model(test_x).argmax(dim=1)
    correct = (preds == test_y).sum().item()
    total = len(test_y)
    acc = correct / total
    print(f"\n候选（CNN）测试集准确率 = {acc:.4f}  ({correct}/{total})")
    print(f"基线准确率              = 0.4261")
    print(f"差值                    = {acc - 0.4261:+.4f}")

    # 4.5) 每类准确率 + 混淆矩阵（判断"哪类最难分"，而不是只看前 10 个失败）
    print("\n每类准确率（测试集内）：")
    for cls in labels:
        idx = label2idx[cls]
        n_cls = int((test_y == idx).sum())
        ok_cls = int(((preds == test_y) & (test_y == idx)).sum())
        print(f"  {cls}: {ok_cls}/{n_cls} = {ok_cls / n_cls:.4f}")

    print("\n混淆矩阵（行=真值，列=预测，顺序同上面类别）：")
    cm = torch.zeros(len(labels), len(labels), dtype=torch.int)
    for i in range(total):
        cm[test_y[i].item(), preds[i].item()] += 1
    for r in range(len(labels)):
        print("  " + " ".join(f"{v:3d}" for v in cm[r].tolist()))

    # 5) 保存结果 + 失败/成功案例
    with open("candidate_result.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc:.4f} correct={correct} total={total}\n")
    failures = [(test_paths[i], test_y[i].item(), preds[i].item())
                for i in range(total) if preds[i].item() != test_y[i].item()]
    successes = [(test_paths[i], test_y[i].item(), preds[i].item())
                 for i in range(total) if preds[i].item() == test_y[i].item()]
    print(f"失败案例数：{len(failures)}（前 10 存 failure_examples.txt）")
    print(f"成功案例数：{len(successes)}（前 10 存 success_examples.txt）")
    with open("failure_examples.txt", "w", encoding="utf-8") as f:
        for path, true, pred in failures[:10]:
            f.write(f"{path}  真值={idx2label[true]}  预测={idx2label[pred]}\n")
    with open("success_examples.txt", "w", encoding="utf-8") as f:
        for path, true, pred in successes[:10]:
            f.write(f"{path}  真值={idx2label[true]}  预测={idx2label[pred]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
