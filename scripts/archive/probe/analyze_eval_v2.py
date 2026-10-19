# -*- coding: utf-8 -*-
"""分析 sft_v2 复测失败模式"""
import json
from collections import Counter

d = json.load(open(r"F:\自研大模型\logs\eval_code_sftv2.json", encoding="utf-8"))
print("summary:", json.dumps(d.get("summary"), ensure_ascii=False))
per = d.get("per_item", [])
if isinstance(per, dict):
    per = list(per.values())
print("per_item 数量:", len(per))

modes = Counter()
for r in per:
    gen = r.get("generation", "") or ""
    if "```python" not in gen or "```" not in gen:
        modes["no_code_block"] += 1
    elif r.get("passed"):
        modes["passed"] += 1
    else:
        modes["code_but_fail"] += 1
print("失败模式:", dict(modes))

# 按 topic 统计
topic_stat = Counter()
for r in per:
    topic_stat[r.get("topic")] += 1
print("topic 分布:", dict(topic_stat))

for r in per[:4]:
    print("=" * 40)
    print("id:", r.get("id"), "| topic:", r.get("topic"), "| passed:", r.get("passed"), "| err:", str(r.get("error"))[:100])
    print("Q:", (r.get("question") or "")[:90].replace("\n", " "))
    gen = (r.get("generation") or "")[:280].replace("\n", " | ")
    print("GEN:", gen)
