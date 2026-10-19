# -*- coding: utf-8 -*-
"""提取全部评测档案的通过率与关键元信息"""
import json, re, os

logs = r"F:\自研大模型\logs"
files = [
    "eval_code.json", "eval_code_base.json", "eval_code_p8.json",
    "eval_code_sftv2.json", "eval_code_sftv3.json", "eval_code_sftv4.json",
    "eval_code_sftv5.json", "eval_code_sftv5_p8.json", "eval_code_sftv5_p16.json",
    "eval_code_sftv6.json", "eval_code_sftv6_p8.json",
    "eval_code_sftv7.json", "eval_code_sftv7_p8.json", "eval_code_sftv7_p16.json",
    "eval_code_sftv8.json", "eval_code_sftv8_p8.json", "eval_code_sftv8_p16.json",
    "eval_code_sftv9.json", "eval_code_sftv10.json", "eval_code_sftv10_p8.json",
]
for f in files:
    p = os.path.join(logs, f)
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, dict):
        meta = {k: v for k, v in d.items() if k != "per_item"}
        n_pass = meta.get("n_pass", meta.get("passed", "?"))
        total = meta.get("total", meta.get("n_items", "?"))
        ckpt = meta.get("ckpt", meta.get("model", "?"))
        k = meta.get("k", 1)
        temp = meta.get("temperature", meta.get("temp", "?"))
        print(f"{f:<30} pass={n_pass}/{total} k={k} temp={temp} ckpt={os.path.basename(str(ckpt))}")
    else:
        print(f"{f:<30} items={len(d)}")
