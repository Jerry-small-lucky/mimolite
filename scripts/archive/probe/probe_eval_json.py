# -*- coding: utf-8 -*-
"""打印 eval_code_sftv2.json 结构"""
import json
d = json.load(open(r"F:\自研大模型\logs\eval_code_sftv2.json", encoding="utf-8"))
print("顶层 keys:", list(d.keys()))
per = d.get("per_item", [])
print("per_item[0] keys:", list(per[0].keys()) if per else "空")
print("per_item[0]:", json.dumps(per[0], ensure_ascii=False)[:500] if per else "")
print("failures 前 2:", json.dumps(d.get("failures", [])[:2], ensure_ascii=False)[:800])
