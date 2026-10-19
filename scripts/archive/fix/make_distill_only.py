# -*- coding: utf-8 -*-
"""试验1：无泄漏纯蒸馏训练集（510 条，剔除评测 100 行）"""
import json
from pathlib import Path

ROOT = Path(r"F:\自研大模型")
meta = json.load(open(ROOT / "logs/eval_set_meta.json", encoding="utf-8"))
eval_lines = set(meta["eval_src_lines"])
old = [json.loads(l) for l in open(ROOT / "data/synth_code.jsonl", encoding="utf-8")]
keep = [d for i, d in enumerate(old) if i not in eval_lines]
out = ROOT / "data/synth_code_distill.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for d in keep:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"无泄漏蒸馏集: {len(keep)} 条 -> {out}")
