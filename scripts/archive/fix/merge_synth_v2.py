# -*- coding: utf-8 -*-
"""数据侧迭代 A：核验现有蒸馏数据 + 合并模板样本 -> synth_code_v2.jsonl（临时工具）"""
import json
from pathlib import Path

ROOT = Path(r"F:\自研大模型")

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

old = load(ROOT / "data/synth_code.jsonl")
tpl = load(ROOT / "data/synth_code_tpl.jsonl")

# 核验现有数据：空字段 / 重复指纹
bad_empty = [d for d in old if not (d.get("question") or "").strip() or not (d.get("code") or "").strip()]
fp = {}
dups = []
for d in old:
    k = (d["question"], d["code"])
    fp.setdefault(k, []).append(d)
dups = [(k, v) for k, v in fp.items() if len(v) > 1]
print(f"现有蒸馏: {len(old)} 条 | 空 question/code: {len(bad_empty)} | 内容重复指纹: {len(dups)}")

# 模板核验
t_bad = [d for d in tpl if not (d.get("question") or "").strip() or not (d.get("code") or "").strip()]
t_fp = len({(d["question"], d["code"]) for d in tpl})
print(f"模板合成: {len(tpl)} 条 | 空字段: {len(t_bad)} | 唯一指纹: {t_fp}/{len(tpl)}")

# 合并
merged = old + tpl
out_p = ROOT / "data/synth_code_v2.jsonl"
with open(out_p, "w", encoding="utf-8") as f:
    for d in merged:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"合并完成: {len(merged)} 条 -> {out_p}")
from collections import Counter
print("src 分布:", dict(Counter(d.get("src") for d in merged)))
print("lang 分布:", dict(Counter(d.get("lang") for d in merged)))
