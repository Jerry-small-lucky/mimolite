# -*- coding: utf-8 -*-
"""统计评测 100 题的函数名（从 tests 提取）"""
import json
import re
from collections import Counter

items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
fn = Counter()
fn_topic = {}
for d in items:
    m = re.search(r"(\w+)\s*\(", d["tests"])
    if m:
        f = m.group(1)
        fn[f] += 1
        fn_topic.setdefault(f, d.get("topic"))
print(f"评测函数名 {len(fn)} 个（100 题）:")
for f, v in fn.most_common():
    print(f"  {v:3d}  {f:24s} {fn_topic[f]}")
