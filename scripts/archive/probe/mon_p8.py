# -*- coding: utf-8 -*-
"""监控 p8 评测完成（60s 轮询，写监控日志）。"""
import json
import time
from pathlib import Path

logs = Path(r"F:\自研大模型\logs")
mon = logs / "mon_p8.txt"
while True:
    he = logs / "std_humaneval_s8.json"
    mb = logs / "std_mbpp_s8.json"
    status = []
    for f in (he, mb):
        if f.exists():
            d = json.load(open(f, encoding="utf-8"))
            status.append(f"{f.name}: {d['summary']['n_pass']}/{d['n']} done")
        else:
            status.append(f"{f.name}: running")
    line = time.strftime("%H:%M:%S") + " | " + " | ".join(status)
    with open(mon, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
    if he.exists() and mb.exists():
        break
    time.sleep(60)
