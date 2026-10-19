# -*- coding: utf-8 -*-
"""核验 v9 模板分布"""
import json
import re
from collections import Counter

lines = [json.loads(l) for l in open(r"F:\自研大模型\data\synth_code_tpl.jsonl", encoding="utf-8")]
fn = Counter()
for d in lines:
    m = re.search(r"def (\w+)\(", d["code"])
    fn[m.group(1)] += 1
print("函数种类:", len(fn))
for f in ["merge", "reverse_linked_list", "count_words", "factorial", "count_characters"]:
    print(f"  {f}: {fn.get(f, 0)}")
print("已删干扰:", {f: fn.get(f, 0) for f in ["count_vowels", "fibonacci", "array_sum", "count_evens"]})
