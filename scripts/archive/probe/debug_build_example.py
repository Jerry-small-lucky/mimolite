# -*- coding: utf-8 -*-
"""调试3：验证 build_example 的 labels 与 assistant token 对齐"""
import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(r"F:\自研大模型")
tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
pad_id = meta["pad"]

def build_example(messages, max_len=512):
    bos = tok.token_to_id("<|bos|>")
    sep = tok.token_to_id("<|sep|>")
    eos = tok.token_to_id("<|eos|>")
    ids = [bos]
    labels = [-100]
    sys_p = tok.encode("用户：").ids
    asst_p = tok.encode("助手：").ids
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    for m in turns:
        content = (m.get("content") or "").strip()
        if m["role"] == "user":
            seg = sys_p + tok.encode(content).ids + tok.encode("\n").ids + [sep] + asst_p
            ids += seg
            labels += [-100] * len(seg)
        else:
            seg = tok.encode(content).ids + [eos]
            ids += seg
            labels += seg
    return ids, labels

lines = [json.loads(l) for l in (ROOT / "data" / "synth_code_sft.jsonl").open(encoding="utf-8") if l.strip()]
d = lines[0]
ids, labels = build_example(d["messages"])
print("总长:", len(ids), "label 非-100 数:", sum(1 for l in labels if l != -100))
# 找到 assistant 起始位置：sep 之后
sep_pos = ids.index(tok.token_to_id("<|sep|>"))
print("sep 位置:", sep_pos)
print("sep 后 tokens:", ids[sep_pos:sep_pos+10])
# assistant 内容 token
asst = d["messages"][1]["content"]
asst_ids = tok.encode(asst).ids
print("assistant 首 10 token:", asst_ids[:10])
print("labels[sep_pos+1:sep_pos+11]:", labels[sep_pos+1:sep_pos+11])
print("对齐正确:", labels[sep_pos+1:sep_pos+1+len(asst_ids)] == asst_ids)
