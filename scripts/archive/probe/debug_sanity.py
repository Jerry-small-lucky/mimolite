# -*- coding: utf-8 -*-
"""调试4：修复后 forward sanity（现有权重跑一小段 loss，仅验证计算路径正常）"""
import json, torch
from pathlib import Path
from tokenizers import Tokenizer
from model import MiMoLite, MiMoLiteConfig

ROOT = Path(r"F:\自研大模型")
tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))

ck = torch.load(ROOT / "runs/sft/final.pt", map_location="cpu", weights_only=False)
model = MiMoLite(MiMoLiteConfig(**ck["config"])).to("cuda")
model.load_state_dict(ck["model"], strict=False)
model.eval()

lines = [json.loads(l) for l in (ROOT / "data" / "synth_code_sft.jsonl").open(encoding="utf-8") if l.strip()]
d = lines[0]
sep = tok.token_to_id("<|sep|>")
ids = [meta["bos"]] + tok.encode(f"用户：{d['messages'][0]['content']}\n").ids + [sep] + tok.encode("助手：").ids + tok.encode(d["messages"][1]["content"]).ids + [meta["eos"]]
x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    out = model(x[:, :-1], labels=x[:, 1:], use_mtp=False)
print("修复后 loss:", out["loss"].item())
