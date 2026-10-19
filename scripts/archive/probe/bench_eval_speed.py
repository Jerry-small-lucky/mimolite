# -*- coding: utf-8 -*-
"""
评测推理速度基准：fp32 vs bf16、batch 1 vs 4
用法：python scripts/bench_eval_speed.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chat import load, build_prompt  # noqa: E402


def bench(model, tok, meta, question, device, bf16: bool, batch: int, max_new: int):
    ids = build_prompt(tok, meta, question)
    x = torch.tensor([ids] * batch, device=device)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    gen = torch.Generator(device="cuda" if device == "cuda" else "cpu")
    gen.manual_seed(2026)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(bf16 and device == "cuda")):
        out = model.generate(x, max_new_tokens=max_new, eos_id=None, generator=gen,
                             temperature=0.2, top_k=50, top_p=0.9,
                             repetition_penalty=1.1, max_context=1024)
    dt = time.time() - t0
    per_item = dt / batch
    print(f"bf16={bf16} batch={batch} | 强制 {max_new} token x{batch} 用时 {dt:.1f}s"
          f" | 每条 {per_item:.1f}s | {max_new / per_item:.1f} tok/s")
    return per_item


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta, ck = load(str(ROOT / "runs/sft/final.pt"), device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M device={device}")
    items = [json.loads(l) for l in (ROOT / "data/eval_code.jsonl").open(encoding="utf-8")
             if l.strip()][:1]
    q = items[0]["question"]
    bench(model, tok, meta, q, device, bf16=False, batch=1, max_new=200)
    bench(model, tok, meta, q, device, bf16=True, batch=1, max_new=200)
    bench(model, tok, meta, q, device, bf16=True, batch=4, max_new=200)


if __name__ == "__main__":
    main()
