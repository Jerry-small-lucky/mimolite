# -*- coding: utf-8 -*-
"""v5 vs v6 pass@1 逐题对比，按函数名/主题统计"""
import json
import re

def load(p):
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("per_item", d.get("results", []))
    return {r["id"]: r for r in d}


def passed(r):
    return r.get("pass", r.get("passed", False))

v5 = load(r"F:\自研大模型\logs\eval_code_sftv8.json")
v6 = load(r"F:\自研大模型\logs\eval_code_sftv10.json")

eval_items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
id_fn = {}
for d in eval_items:
    m = re.search(r"(\w+)\s*\(", d["tests"])
    id_fn[d["id"]] = (m.group(1) if m else "?", d.get("topic"))

improved, worsened, same = [], [], []
for d in eval_items:
    i = d["id"]
    a = passed(v5[i])
    b = passed(v6[i])
    if b and not a:
        improved.append(i)
    elif a and not b:
        worsened.append(i)
    else:
        same.append(i)

print(f"v5={sum(passed(v) for v in v5.values())}/100  v6={sum(passed(v) for v in v6.values())}/100")
print(f"\nv6 新通过 {len(improved)} 题（v5 未过）:")
for i in improved:
    fn, tp = id_fn[i]
    print(f"  {i}  {fn} ({tp})")
print(f"\nv6 回退 {len(worsened)} 题（v5 过 v6 未过）:")
for i in worsened:
    fn, tp = id_fn[i]
    print(f"  {i}  {fn} ({tp})")

# 按函数名聚合
from collections import Counter
def by_fn(ds):
    c = Counter()
    t = Counter()
    for i, v in ds.items():
        fn, tp = id_fn[i]
        c[(fn, tp)] += 1 if passed(v) else 0
        t[(fn, tp)] += 1
    return c, t
c5, t5 = by_fn(v5)
c6, t6 = by_fn(v6)
print(f"\n{'函数':26s} {'题数':>4s} {'v5':>4s} {'v6':>4s}")
for (fn, tp), n in sorted(t5.items(), key=lambda x: -x[1]):
    print(f"{fn:26s} {n:4d} {c5[(fn,tp)]:4d} {c6[(fn,tp)]:4d}  ({tp})")
