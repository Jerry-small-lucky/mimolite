# -*- coding: utf-8 -*-
"""sft_v3 失败模式快速抽查"""
import json
d = json.load(open(r"F:\自研大模型\logs\eval_code_sftv3.json", encoding="utf-8"))
print("summary:", d.get("summary"))
per = d.get("per_item", [])
from collections import Counter
c = Counter(str(r.get("reason"))[:55] for r in per)
for k, v in c.most_common(8):
    print(f"  {v:3d}  {k}")
for f in d.get("failures", [])[:3]:
    print("=" * 40)
    print("id:", f.get("id"), "| reason:", str(f.get("reason"))[:70])
    print("Q:", (f.get("question") or "")[:70].replace("\n", " "))
    print("GEN:", (f.get("gen_excerpt") or "")[:180].replace("\n", " | "))
