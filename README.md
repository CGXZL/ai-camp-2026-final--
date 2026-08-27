# 垃圾分类识别（可回收 / 厨余 / 有害 / 其他）

为回收站分类员做的垃圾图像分类工具：输入一张垃圾照片，输出可回收/厨余/有害/其他四类之一。

## 数据

- 来源：Kaggle **Garbage classification dataset**（shanlan426），许可 **Apache 2.0**（教师已批准）
- 原始 URL：https://www.kaggle.com/datasets/shanlan426/garbage-classification-dataset
- 内容：55 个细类文件夹（命名 `大类_细类`），本项目按前缀合并成 **4 大类**：厨余垃圾 5153 / 可回收物 3331 / 有害垃圾 2158 / 其他垃圾 1452
- 下载：登录 Kaggle → Download → 解压，把解压出的 `dataset` 文件夹放到 `data/raw/` 下（即 `data/raw/dataset/厨余垃圾_哈密瓜/...`）
- 大文件已被 `.gitignore` 排除，不会提交到 GitHub

## 环境

- Python 3 + PyTorch + Pillow
- 安装：`pip install torch pillow`

## 运行（按顺序）

```powershell
python make_split.py          # 1) 固定 80/20 划分，生成 split.json
python baseline4.py           # 2) 基线（多数类）
python train_candidate4.py    # 3) 候选一（小 CNN），约几分钟
python candidate_v2.py        # 4) 候选二（迁移学习，最终候选），约几分钟，保存 model_v2.pt
python predict.py 图片路径    # 5) 现场演示：预测一张新照片属于哪类
```

预期输出：
- `baseline4.py` → 准确率 **0.4261**
- `train_candidate4.py` → 准确率 **0.6268**
- `candidate_v2.py` → 准确率 **0.8858**，并保存 `model_v2.pt`
- `predict.py` → 打印照片的预测类别和置信度

## 测试

```powershell
python -m pytest tests/
```

检查 `split.json` 的数据契约：4 大类都存在、每类数量在预期范围。当前结果：2 passed。

## 结果摘要

- 基线 0.4261 vs 候选一（小 CNN）0.6268 vs 候选二（迁移学习）**0.8858**（同一测试集 2417 张）
- 每类准确率（候选二）：厨余 0.96 / 有害 0.89 / 其他 0.81 / 可回收 0.80
- 关键发现：从零训练的小 CNN 中可回收物最弱（0.39）；迁移学习把每类准确率都大幅提升
- 成功与失败案例见 `report.md`

## 限制

数据：白底室内拍摄、类别不均衡：厨余最多，其他最少；
评估：总准确率被"厨余"拉高，必须看每类
产品：只适用白底照片，不能自动决策

