# -*- coding: utf-8 -*-
"""核验加权分配：hot 函数 vs 非 hot"""
import json
import re
from collections import Counter

lines = [json.loads(l) for l in open(r"F:\自研大模型\data\synth_code_tpl.jsonl", encoding="utf-8")]
fn = Counter()
for d in lines:
    m = re.search(r"def (\w+)\(", d["code"])
    fn[m.group(1)] += 1
hot = {"factorial", "count_characters", "count_ones", "count_chars", "process_string",
       "count_elements", "fib", "merge", "reverse_linked_list", "count_words", "sum_array",
       "sum_even_numbers", "convert_to_ascii", "count_set_bits", "factor_sum"}
print("HOT 函数条数:")
for f in sorted(hot, key=lambda x: -fn.get(x, 0)):
    print(f"  {fn.get(f, 0):3d}  {f}")
nonhot = {k: v for k, v in fn.items() if k not in hot}
print(f"非 hot 函数数: {len(nonhot)}，平均 {sum(nonhot.values())/len(nonhot):.1f} 条/函数")
