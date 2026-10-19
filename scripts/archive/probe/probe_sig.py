# -*- coding: utf-8 -*-
"""诊断：函数签名提示对生成的影响（sft_v2 权重，2 题对比）"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"F:\自研大模型")
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from model import MiMoLite, MiMoLiteConfig

CKPT = ROOT / "runs" / "sft_v2" / "final.pt"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = MiMoLiteConfig(**ck["config"])
tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
model = MiMoLite(cfg).cuda().eval()
model.load_state_dict(ck["model"], strict=False)

items = [json.loads(l) for l in (ROOT / "data" / "eval_code.jsonl").open(encoding="utf-8")]

def fn_from_tests(tests):
    m = re.search(r"(\w+)\s*\(", tests)
    return m.group(1) if m else None

def generate(prompt, max_new=256, temp=0.2):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    gen = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x[:, -cfg.window:])["logits"][:, -1, :]
            if temp > 0:
                probs = torch.softmax(logits / temp, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            g = nxt.item()
            gen.append(g)
            if g == meta["eos"]:
                break
            x = torch.cat([x, nxt], dim=1)
    return tok.decode(gen)

for it in items[:2]:
    q = it["question"]
    fn = fn_from_tests(it["tests"])
    print("=" * 50)
    print(f"[{it['id']}] topic={it.get('topic')}  fn={fn}")
    print("Q:", q[:90].replace("\n", " "))
    p0 = q
    p1 = f"{q}\n请实现函数 {fn}。"
    g0 = generate(p0)
    g1 = generate(p1)
    print("--- 无签名 ---")
    print(g0[:180].replace("\n", " | "))
    print("--- 带签名 ---")
    print(g1[:180].replace("\n", " | "))
