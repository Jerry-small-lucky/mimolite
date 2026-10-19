# -*- coding: utf-8 -*-
"""打印评测未覆盖主题的题目+tests 格式"""
import json

items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
missing = {"链表基础操作", "回文检测", "数列与递推"}
for d in items:
    if d.get("topic") in missing:
        print("=" * 50)
        print(f"id={d['id']} topic={d.get('topic')} lang={d.get('lang')}")
        print("Q:", d["question"])
        print("T:", d["tests"])
