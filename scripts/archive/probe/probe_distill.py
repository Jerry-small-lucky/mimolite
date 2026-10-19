# -*- coding: utf-8 -*-
"""检查 distill 数据性质与模板样例"""
import json

p = r"F:\自研大模型\data\synth_code_distill.jsonl"
lines = [json.loads(l) for l in open(p, encoding="utf-8")]
print("条数:", len(lines))
print("字段:", list(lines[0].keys()))
print("样例0:", json.dumps(lines[0], ensure_ascii=False)[:400])
