"""把 55 个细类按前缀合并成 4 大类，并做一个固定的 80/20 训练/测试划分。

- 合并规则：文件夹名如"厨余垃圾_哈密瓜"，下划线前的"厨余垃圾"就是大类。
- 划分：每个大类内部随机 80% 训练、20% 测试；用 SEED 固定，保证可复现。
- 结果：保存 split.json，基线和候选都读它，确保用同一个划分（同条件比较）。
"""

import json
import random
from pathlib import Path

DATA_ROOT = Path("data/raw/dataset")
SEED = 42
TEST_RATIO = 0.2

random.seed(SEED)


def main() -> int:
    # 1) 把每个细类文件夹的图片，按前缀归到大类
    classes = {}   # 大类名 -> [图片路径, ...]
    for folder in DATA_ROOT.iterdir():
        if not folder.is_dir():
            continue
        coarse = folder.name.split("_")[0]          # 取前缀作大类
        images = [p for p in folder.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        classes.setdefault(coarse, []).extend(images)

    # 2) 每个大类内部，按固定种子做 80/20 划分
    split = {"TRAIN": {}, "TEST": {}}
    for coarse, images in sorted(classes.items()):
        random.shuffle(images)                     # 打乱（种子固定 → 每次一样）
        n_test = int(len(images) * TEST_RATIO)
        test = images[:n_test]
        train = images[n_test:]
        split["TRAIN"][coarse] = [str(p) for p in train]
        split["TEST"][coarse] = [str(p) for p in test]
        print(f"{coarse}: 共 {len(images)} → 训练 {len(train)} / 测试 {len(test)}")

    # 3) 保存划分，供 baseline 和候选共用
    with open("split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print("已保存 split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
