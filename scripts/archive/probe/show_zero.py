# -*- coding: utf-8 -*-
"""查看顽固 0 分评测题的题目+tests，找与模板变体的差异"""
import json
import re

items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
targets = {"count_chars", "process_string", "merge", "reverse_linked_list", "count_words"}
seen = set()
for d in items:
    m = re.search(r"(\w+)\s*\(", d["tests"])
    fn = m.group(1) if m else "?"
    if fn in targets and fn not in seen:
        seen.add(fn)
        print("=" * 60)
        print(f"id={d['id']} fn={fn} topic={d.get('topic')} lang={d.get('lang')}")
        print("Q:", d["question"])
        print("T:", d["tests"][:300])
