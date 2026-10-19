"""
用 PyTorch 直接对话（无需先转 GGUF，方便快速验证各阶段权重）
================================================================

用法：
  python chat.py --ckpt runs/pretrain/final.pt --prompt "介绍一下深度学习"
  python chat.py --ckpt runs/sft/final.pt --interactive
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(ckpt_path: str, device: str):
    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    model = MiMoLite(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = ck.get("tokenizer_meta") or json.loads(
        (ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    return model, tok, meta, ck


def build_prompt(tok, meta, user: str) -> list[int]:
    sep = meta.get("sep") if isinstance(meta.get("sep"), int) else tok.token_to_id("<|sep|>")
    text = f"用户：{user}\n"
    return [meta["bos"]] + tok.encode(text).ids + [sep] + tok.encode("助手：").ids


def reply(model, tok, meta, user: str, max_new: int = 128,
          temperature: float = 0.8, device="cuda") -> str:
    ids = build_prompt(tok, meta, user)
    x = torch.tensor([ids], device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=temperature,
                         top_k=50, top_p=0.9, eos_id=meta["eos"],
                         repetition_penalty=1.1, max_context=1024)
    gen = out[0][len(ids):].tolist()
    if meta["eos"] in gen:
        gen = gen[:gen.index(meta["eos"])]
    return tok.decode(gen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/pretrain/final.pt"))
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta, ck = load(args.ckpt, device)
    log(f"载入 {args.ckpt} (step={ck.get('step')}), "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M, device={device}")

    prompts = [args.prompt] if args.prompt else []
    if args.interactive:
        log("进入交互模式，输入 exit 退出")
        while True:
            try:
                q = input("\n你> ").strip()
            except EOFError:
                break
            if q.lower() in ("exit", "quit", "退出"):
                break
            if not q:
                continue
            t0 = time.time()
            r = reply(model, tok, meta, q, args.max_new, args.temp, device)
            print(f"模型> {r}   [{time.time()-t0:.1f}s]")
    for q in prompts:
        t0 = time.time()
        r = reply(model, tok, meta, q, args.max_new, args.temp, device)
        print(f"\n你> {q}\n模型> {r}   [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()