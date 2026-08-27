"""最小数据契约测试：确认 split.json 结构正确、4 大类数量在预期范围。

运行：python -m pytest tests/
"""

import json
from pathlib import Path

SPLIT_FILE = Path("split.json")

EXPECTED_CLASSES = ["其他垃圾", "厨余垃圾", "可回收物", "有害垃圾"]


def _load():
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def test_split_has_expected_four_classes():
    data = _load()
    assert set(data["TRAIN"].keys()) == set(EXPECTED_CLASSES)
    assert set(data["TEST"].keys()) == set(EXPECTED_CLASSES)


def test_split_sizes_are_reasonable():
    data = _load()
    for cls in EXPECTED_CLASSES:
        assert len(data["TRAIN"][cls]) > 500   # 每类训练至少几百张
        assert len(data["TEST"][cls]) > 200    # 每类测试至少 200 张
    total_test = sum(len(v) for v in data["TEST"].values())
    assert 2000 < total_test < 3000            # 测试集总数约 2417
