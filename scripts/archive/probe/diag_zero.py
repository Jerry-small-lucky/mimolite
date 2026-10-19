# -*- coding: utf-8 -*-
"""诊断：count_chars/count_characters/count_words/merge/reverse_linked_list 全部题面"""
import json
import re

items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
targets = {"count_chars", "count_characters", "count_words", "merge", "reverse_linked_list"}
for d in items:
    m = re.search(r"(\w+)\s*\(", d["tests"])
    fn = m.group(1) if m else "?"
    if fn in targets:
        print("=" * 70)
        print(f"id={d['id']} fn={fn} topic={d.get('topic')} lang={d.get('lang')}")
        print("Q:", d["question"])
