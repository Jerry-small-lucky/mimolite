# -*- coding: utf-8 -*-
"""函数级 pass@1 vs pass@8 对比分析（v8）：
证明数据侧价值的核心证据——哪些函数 pass@1=0 但 pass@8 多采样命中，
即模型已学到能力、只是单采样确定性不足；以及 pass@1 已稳/双零死题。
输出：logs/fn_v8_matrix.md（markdown 表格）+ 控制台摘要
"""
import json
import re
from collections import defaultdict

EVAL = r"F:\自研大模型\data\eval_code.jsonl"
P1 = r"F:\自研大模型\logs\eval_code_sftv8.json"
P8 = r"F:\自研大模型\logs\eval_code_sftv8_p8.json"
OUT = r"F:\自研大模型\logs\fn_v8_matrix.md"

FN_RE = re.compile(r"assert\s+([a-zA-Z_]\w*)\(")


def func_of(tests: str) -> str:
    m = FN_RE.search(tests)
    return m.group(1) if m else "?"


rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
id2row = {r["id"]: r for r in rows}

p1 = json.load(open(P1, encoding="utf-8"))["per_item"]
p8 = json.load(open(P8, encoding="utf-8"))["per_item"]
p1_by_id = {x["id"]: x["pass"] for x in p1}
p8_by_id = {x["id"]: x["pass"] for x in p8}

# 按函数名聚合
agg = defaultdict(lambda: {"n": 0, "p1": 0, "p8": 0, "ids": []})
for r in rows:
    fn = func_of(r["tests"])
    a = agg[fn]
    a["n"] += 1
    a["ids"].append(r["id"])
    if p1_by_id.get(r["id"]):
        a["p1"] += 1
    if p8_by_id.get(r["id"]):
        a["p8"] += 1

lines = ["# v8 函数级 pass@1 vs pass@8 对比（评测 100 题）\n"]
lines.append("| 函数 | 题数 | pass@1 通过 | pass@8 通过 | 类型 |")
lines.append("|---|---:|---:|---:|---|")

types = {"ms": [], "stable": [], "partial": [], "dead": []}
for fn, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
    if a["p1"] == a["n"]:
        t = "已稳（p1 全过）"
        types["stable"].append((fn, a))
    elif a["p1"] == 0 and a["p8"] > 0:
        t = "多采样命中（p1=0→p8>0）"
        types["ms"].append((fn, a))
    elif 0 < a["p1"] < a["n"] and a["p8"] >= a["n"]:
        t = "p8 补全（p8 全过）"
        types["partial"].append((fn, a))
    elif a["p1"] == 0 and a["p8"] == 0:
        t = "双零死题"
        types["dead"].append((fn, a))
    else:
        t = "部分通过"
        types["partial"].append((fn, a))
    lines.append(f"| {fn} | {a['n']} | {a['p1']} | {a['p8']} | {t} |")

lines.append("")
lines.append(f"## 汇总：稳定 {len(types['stable'])} 个函数 / 多采样命中 {len(types['ms'])} 个 / 部分 {len(types['partial'])} 个 / 双零死题 {len(types['dead'])} 个")
lines.append("")
lines.append("### 多采样命中型（模型已学到，单采样确定性不足——数据侧价值的直接证据）")
lines.append("| 函数 | 题数 | p8 命中 |")
lines.append("|---|---:|---:|")
for fn, a in types["ms"]:
    lines.append(f"| {fn} | {a['n']} | {a['p8']} |")
lines.append("")
lines.append("### 双零死题（能力未达 / 评测设计问题）")
lines.append("| 函数 | 题数 |")
lines.append("|---|---:|")
for fn, a in types["dead"]:
    lines.append(f"| {fn} | {a['n']} |")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

ms_ids = [i for fn, a in types["ms"] for i in a["ids"]]
print(f"函数总数 {len(agg)} | 稳定 {len(types['stable'])} | 多采样命中 {len(types['ms'])} | 部分 {len(types['partial'])} | 死题 {len(types['dead'])}")
print("多采样命中函数:", [fn for fn, _ in types["ms"]])
print("死题函数:", [fn for fn, _ in types["dead"]])
print("输出:", OUT)
