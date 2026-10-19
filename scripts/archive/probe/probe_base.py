# -*- coding: utf-8 -*-
"""确认基座对照与早期评测摘要"""
import json

for f in ["eval_code_base.json", "eval_code.json", "eval_code_p8.json"]:
    d = json.load(open(r"F:\自研大模型\logs\\" + f, encoding="utf-8"))
    s = d.get("summary", {})
    meta = d.get("meta", {})
    print(f, "| pass:", s.get("n_pass"), "/", s.get("n"), "| rate:", s.get("pass_rate"),
          "| ckpt:", meta.get("ckpt", "").split("\\")[-1])
