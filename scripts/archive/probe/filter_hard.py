# -*- coding: utf-8 -*-
"""从 grpo_train 筛出 v8 通过率=0 的最难题（信号最密）"""
import json

src = r"F:\自研大模型\data\grpo_train.jsonl"
out = r"F:\自研大模型\data\grpo_hard.jsonl"
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
hard = [r for r in rows if r.get("pass_rate", 1) == 0.0]
with open(out, "w", encoding="utf-8") as f:
    for r in hard:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"grpo_train {len(rows)} -> hard {len(hard)}")
