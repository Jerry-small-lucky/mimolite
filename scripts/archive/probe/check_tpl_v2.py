# -*- coding: utf-8 -*-
"""核验 v2 模板：变体数、每函数条数分布、主题覆盖、泄漏校验"""
import json
import re
from collections import Counter

tpl = [json.loads(l) for l in open(r"F:\自研大模型\data\synth_code_tpl.jsonl", encoding="utf-8")]
print(f"模板总数: {len(tpl)}")
fn = Counter()
for d in tpl:
    m = re.search(r"def (\w+)\(", d["code"])
    fn[m.group(1)] += 1
print(f"函数名种类: {len(fn)}")
top = fn.most_common(20)
print("每函数条数 Top20（条数/函数）:")
for f, v in top:
    print(f"  {v:3d}  {f}")
print("条数>6 的函数:", [f for f, v in fn.items() if v > 6])

topic = Counter(d["topic"] for d in tpl)
print("\n主题分布:", dict(topic))

# 与评测函数名对齐率
eval_items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
eval_fn = set()
for d in eval_items:
    m = re.search(r"(\w+)\s*\(", d["tests"])
    if m:
        eval_fn.add(m.group(1))
overlap = eval_fn & set(fn)
print(f"\n评测 {len(eval_fn)} 个函数名中，模板覆盖 {len(overlap)} 个")
print("未覆盖的评测函数名:", sorted(eval_fn - set(fn)))
