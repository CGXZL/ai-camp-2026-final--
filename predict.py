"""用训练好的迁移学习模型预测一张新图片属于哪类垃圾（现场演示用）。

用法：
    python predict.py 图片路径

需要先运行 python candidate_v2.py 生成 model_v2.pt（预训练 ResNet18 + 线性分类器）。
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

SPLIT_FILE = Path("split.json")
IMG_SIZE = 224

# 和 candidate_v2.py 一样的预处理（ImageNet 归一化）
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python predict.py 图片路径")
        return 1
    img_path = Path(sys.argv[1])
    if not img_path.is_file():
        print(f"找不到图片：{img_path}")
        return 1

    # 类别顺序和训练时一致
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    labels = sorted(split["TRAIN"].keys())
    idx2label = {i: name for i, name in enumerate(labels)}

    # 特征提取器：预训练 ResNet18（和 candidate_v2.py 一样，去掉分类头）
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # 加载训练好的线性分类器（512 → 4）
    model = nn.Linear(512, len(labels))
    model.load_state_dict(torch.load("model_v2.pt", map_location="cpu"))
    model.eval()

    # 预处理 → 特征 → 预测
    x = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = backbone(x)
        out = model(feat)
        prob = torch.softmax(out, dim=1)[0]
        pred = int(out.argmax(dim=1).item())

    print(f"预测类别：{idx2label[pred]}（置信度 {prob[pred].item():.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
