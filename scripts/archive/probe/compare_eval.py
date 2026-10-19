# -*- coding: utf-8 -*-
"""对比旧/新 SFT 评测失败模式 + 手动生成诊断"""
import json
from collections import Counter

def load_reasons(p):
    d = json.load(open(p, encoding="utf-8"))
    per = d.get("per_item", [])
    if isinstance(per, dict):
        per = list(per.values())
    return Counter(r.get("reason", "?") for r in per), d.get("summary", {})

for name, p in [("旧SFT(泄漏基线)", r"F:\自研大模型\logs\eval_code.json"),
                ("新SFT(sft_v2无泄漏)", r"F:\自研大模型\logs\eval_code_sftv2.json")]:
    try:
        c, s = load_reasons(p)
        print(f"[{name}] pass_rate={s.get('pass_rate')} reasons={dict(c)}")
    except Exception as e:
        print(f"[{name}] 读取失败: {e}")

# 旧模型失败样例（看旧模型代码块质量）
d = json.load(open(r"F:\自研大模型\logs\eval_code.json", encoding="utf-8"))
fails = d.get("failures", [])
print("\n旧模型失败样例 2 条:")
for f in fails[:2]:
    print("id:", f.get("id"), "| reason:", f.get("reason"))
    print("Q:", (f.get("question") or "")[:80].replace("\n", " "))
    print("GEN:", (f.get("gen_excerpt") or "")[:200].replace("\n", " | "))
