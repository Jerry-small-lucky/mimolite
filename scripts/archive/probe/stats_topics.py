# -*- coding: utf-8 -*-
"""统计评测集 topic 分布 + 已覆盖对照"""
import json
from collections import Counter

items = [json.loads(l) for l in open(r"F:\自研大模型\data\eval_code.jsonl", encoding="utf-8")]
c = Counter()
for d in items:
    c[d.get("topic")] += 1
print("评测 100 题 topic 分布:")
for k, v in c.most_common():
    print(f"  {v:3d}  {k}")

# 模板当前主题
print("\n模板当前 9 主题: 哈希表计数 / 排序与比较 / 字符串处理 / 简单数学计算 / 数组遍历与统计 / 双指针 / 递归基础 / 位运算基础 / 贪心策略")
covered = {"哈希表计数", "排序与比较", "字符串处理", "简单数学计算", "数组遍历与统计", "双指针", "递归基础", "位运算基础", "贪心策略"}
missing = [k for k in c if k not in covered]
print(f"\n评测中模板未覆盖的 topic: {missing}")
