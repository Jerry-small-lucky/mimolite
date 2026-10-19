# -*- coding: utf-8 -*-
"""
筛选 GRPO 训练数据：用当前最佳模型对新合成题采样，选通过率低的题。
RL 信号来自"模型答不好的可验证题"；模板题对 v8 过拟合，无学习信号。
输出：data/grpo_train.jsonl（筛选后的 RL 训练集）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval_code import load_model, generate_batch, extract_code, run_sandbox  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/sft_v8/final.pt"))
    ap.add_argument("--data", type=str, default=str(ROOT / "data/synth_code_tpl_seed8.jsonl"))
    ap.add_argument("--out", type=str, default=str(ROOT / "data/grpo_train.jsonl"))
    ap.add_argument("--samples", type=int, default=2, help="每题采样次数")
    ap.add_argument("--thresh", type=float, default=0.5, help="通过率<=阈值的题入选")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta, ck = load_model(args.ckpt, device)
    items = [json.loads(l) for l in Path(args.data).open(encoding="utf-8") if l.strip()]
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    log(f"载入 {args.ckpt} | 候选题 {len(items)} 条 | 采样 {args.samples}x | 阈值 {args.thresh}")

    sel, drop = [], 0
    t0 = time.time()
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        qs = [c["question"] for c in chunk]
        ok_counts = [0] * len(chunk)
        for s in range(args.samples):
            texts_lens = generate_batch(model, tok, meta, qs, device,
                                        args.max_new, args.seed + s,
                                        bf16=(device == "cuda"))
            for i, (text, _) in enumerate(texts_lens):
                code = extract_code(text)
                if code is None:
                    continue
                ok, _ = run_sandbox(code, chunk[i]["tests"], args.timeout)
                if ok:
                    ok_counts[i] += 1
        for i, c in enumerate(chunk):
            rate = ok_counts[i] / args.samples
            if rate <= args.thresh:
                c["pass_rate"] = rate
                sel.append(c)
            else:
                drop += 1
        if (start + args.batch) % 50 == 0 or start + args.batch >= len(items):
            el = time.time() - t0
            log(f"{min(start+args.batch, len(items))}/{len(items)} 题 | 已入选 {len(sel)} "
                f"({el:.0f}s)")

    with open(args.out, "w", encoding="utf-8") as f:
        for c in sel:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log(f"完成：入选 {len(sel)} 题（通过率<= {args.thresh}），淘汰 {drop} -> {args.out}")


if __name__ == "__main__":
    main()
