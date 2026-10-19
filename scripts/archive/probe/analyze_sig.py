# -*- coding: utf-8 -*-
"""分析 sig 档案失败模式 + 抽样本"""
import json
from collections import Counter

d = json.load(open(r"F:\自研大模型\logs\eval_code_sftv2_sig.json", encoding="utf-8"))
per = d.get("per_item", [])
if isinstance(per, dict):
    per = list(per.values())
print("summary:", d.get("summary"))
print("reason 分布:")
c = Counter()
for r in per:
    c[str(r.get("reason"))[:60]] += 1
for k, v in c.most_common(15):
    print(f"  {v:3d}  {k}")

fails = d.get("failures", [])
print("\n失败样例 3 条:")
for f in fails[:3]:
    print("=" * 40)
    print("id:", f.get("id"), "| reason:", str(f.get("reason"))[:80])
    print("Q:", (f.get("question") or "")[:80].replace("\n", " "))
    print("GEN:", (f.get("gen_excerpt") or "")[:220].replace("\n", " | "))
