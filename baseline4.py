"""4 类多数类基线：把训练集里数量最多的类别，当作所有测试图片的预测结果。

读取 make_split.py 生成的 split.json——基线和候选用同一份划分（同条件比较）。
"""

import json
from pathlib import Path

SPLIT_FILE = Path("split.json")


def main() -> int:
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)

    train = split["TRAIN"]
    test = split["TEST"]

    # 1) 看训练集：哪一类图片最多，谁就是"多数类"
    train_sizes = {label: len(paths) for label, paths in train.items()}
    majority = max(train_sizes, key=train_sizes.get)
    print("训练集各类数量:", train_sizes)
    print(f"多数类是 {majority}")

    # 2) 测试集：每类张数
    test_sizes = {label: len(paths) for label, paths in test.items()}
    total = sum(test_sizes.values())
    correct = test_sizes[majority]
    accuracy = correct / total
    print("测试集各类数量:", test_sizes)
    print(f"基线（全部预测为 {majority}）：正确 {correct} 张 / {total}")
    print(f"准确率 = {accuracy:.4f}")

    # 3) 保存，供候选比较
    with open("baseline_result.txt", "w", encoding="utf-8") as f:
        f.write(f"majority={majority} correct={correct} total={total} accuracy={accuracy:.4f}\n")
    print("已保存 baseline_result.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
