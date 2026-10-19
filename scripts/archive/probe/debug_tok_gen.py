# -*- coding: utf-8 -*-
"""调试2：多温度生成测试"""
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

q = "统计字符串中每个字符出现的次数，并返回一个字典。"
sep = tok.token_to_id("<|sep|>")
ids = [meta["bos"]] + tok.encode(f"用户：{q}\n").ids + [sep] + tok.encode("助手：").ids
x = torch.tensor([ids], device="cuda")
for temp in (0.0, 0.5, 0.9):
    out = model.generate(x, max_new_tokens=120, temperature=temp, top_k=50, top_p=0.9,
                         eos_id=meta["eos"], repetition_penalty=1.1, max_context=1024)
    gen = out[0][len(ids):].tolist()
    if meta["eos"] in gen:
        gen = gen[:gen.index(meta["eos"])]
    print(f"--- temp={temp} 生成 {len(gen)} tokens ---")
    print(repr(tok.decode(gen)[:300]))
