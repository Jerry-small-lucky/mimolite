"""
阶段二：SFT 指令微调 —— 让 Base 模型学会对话
============================================

对应小米 MiMo 后训练范式中的第一阶段：
  MiMo-V2-Flash / MiMo-V2.5-Pro 的三阶段后训练 =
      ① 监督微调 SFT（建立基础指令跟随）
      ② 领域专精训练（各领域 teacher 做 RL）
      ③ 多教师在线策略蒸馏 MOPD
  本机实现 ①（全量微调），③ 的简化版见 grpo.py 的教师蒸馏分支。

对话模板：
  <|bos|>用户：{内容}<|sep|>助手：{回复}<|eos|>
  损失只在"助手回复 + <|eos|>"上计算（prompt 部分置 -100）

用法：
  python sft.py --epochs 2 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_example(tok, messages: list[dict], meta: dict, max_len: int):
    """返回 (input_ids, labels)；prompt 部分 label 置 -100。"""
    bos = tok.token_to_id("<|bos|>")
    sep = tok.token_to_id("<|sep|>")
    eos = tok.token_to_id("<|eos|>")

    ids: list[int] = [bos]
    labels: list[int] = [-100]

    sys_p = tok.encode("用户：").ids
    asst_p = tok.encode("助手：").ids

    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) < 2:
        return None

    for i, m in enumerate(turns):
        content = (m.get("content") or "").strip()
        if not content:
            return None
        if m["role"] == "user":
            seg = sys_p + tok.encode(content).ids + tok.encode("\n").ids + [sep] + asst_p
            ids += seg
            labels += [-100] * len(seg)          # prompt 不计损失
        else:
            seg = tok.encode(content).ids + [eos]
            ids += seg
            labels += seg                        # 回复计损失

    if len(ids) > max_len:
        return None
    if not any(l != -100 for l in labels):
        return None
    return np.asarray(ids, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def collate(batch, pad_id: int):
    n = max(len(x[0]) for x in batch)
    x = np.full((len(batch), n), pad_id, dtype=np.int64)
    y = np.full((len(batch), n), -100, dtype=np.int64)
    m = np.zeros((len(batch), n), dtype=np.float32)
    for i, (ids, lab) in enumerate(batch):
        x[i, :len(ids)] = ids
        y[i, :len(lab)] = lab
        m[i, :len(ids)] = 1.0
    return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(RUNS / "pretrain/final.pt"))
    ap.add_argument("--data", type=str, default=str(DATA / "sft.jsonl"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--run-name", type=str, default="sft")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--val-n", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    if not Path(args.ckpt).exists():
        log(f"找不到基座 {args.ckpt}，请先完成预训练")
        return

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    pad_id = meta["pad"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 数据 ----
    log(f"读取 SFT 数据 {args.data}")
    raw = [json.loads(l) for l in Path(args.data).open(encoding="utf-8")]
    if args.max_samples:
        raw = raw[:args.max_samples]
    examples = []
    for r in raw:
        ex = build_example(tok, r["messages"], meta, args.max_len)
        if ex:
            examples.append(ex)
    log(f"可用样本 {len(examples):,} / {len(raw):,}（超长或格式不符被丢弃）")
    if not examples:
        return

    random.Random(42).shuffle(examples)
    val = examples[:args.val_n]
    train = examples[args.val_n:]
    # 长度分桶，减少 padding 浪费
    train.sort(key=lambda e: len(e[0]) // 64)
    buckets = [train[i:i + args.micro_bs] for i in range(0, len(train), args.micro_bs)]

    micro = MiMoLite(cfg).to(device)
    micro.load_state_dict(ck["model"], strict=False)
    log(f"载入基座完成，params={sum(p.numel() for p in micro.parameters())/1e6:.1f}M")

    decay, no_decay = [], []
    for n_, p in micro.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    run_dir = RUNS / args.run_name
    run_dir.mkdir(exist_ok=True)

    steps_per_epoch = max(1, len(buckets) // args.accum)
    total_steps = int(steps_per_epoch * args.epochs)
    log(f"训练配置：{total_steps} steps (每 epoch {steps_per_epoch} steps), "
        f"micro_bs={args.micro_bs} accum={args.accum}")

    @torch.no_grad()
    def evaluate() -> float:
        micro.eval()
        vbuckets = [val[i:i + args.micro_bs] for i in range(0, len(val), args.micro_bs)]
        tot, ntok = 0.0, 0
        for b in vbuckets:
            x, y, m = collate(b, pad_id)
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = micro(x[:, :-1], labels=y[:, 1:])
            n = int((y[:, 1:] != -100).sum())
            tot += out["loss"].item() * n
            ntok += n
        micro.train()
        return tot / max(ntok, 1)

    v0 = evaluate()
    log(f"初始验证 loss = {v0:.4f}")

    micro.train()
    step = 0
    t0 = time.time()
    hist = []
    ema = None
    done = False

    for epoch in range(math.ceil(args.epochs)):
        random.Random(42 + epoch).shuffle(buckets)
        for i, b in enumerate(buckets):
            x, y, m = collate(b, pad_id)
            x, y, m = x.to(device), y.to(device), m.to(device)

            # 学习率调度
            if step < args.warmup:
                lr = args.lr * (step + 1) / args.warmup
            else:
                prog = (step - args.warmup) / max(1, total_steps - args.warmup)
                prog = min(1.0, prog)
                lr = args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio)
                                * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = micro(x[:, :-1], labels=y[:, 1:])
            loss = out["loss"]
            (loss / args.accum).backward()

            if (i + 1) % args.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(micro.parameters(), args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                lv = loss.item() / args.accum
                ema = lv if ema is None else 0.9 * ema + 0.1 * lv
                hist.append({"step": step, "loss": lv, "lr": lr})

                if step % args.log_every == 0:
                    el = time.time() - t0
                    log(f"ep{epoch} step {step}/{total_steps} | loss {lv:.4f} (ema {ema:.4f}) | "
                        f"lr {lr:.2e} | gnorm {float(gn):.2f} | {el/60:.1f}m")
                if step % 500 == 0:
                    vl = evaluate()
                    log(f"  >>> step {step} val loss = {vl:.4f} (初始 {v0:.4f})")
                    torch.save({"model": micro.state_dict(), "config": ck["config"],
                                "tokenizer_meta": meta, "step": step, "sft_v0": v0},
                               run_dir / "last.pt")
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    vl = evaluate()
    log(f"最终验证 loss = {vl:.4f}  (初始 {v0:.4f})")
    torch.save({"model": micro.state_dict(), "config": ck["config"],
                "tokenizer_meta": meta, "step": step,
                "val_before": v0, "val_after": vl}, run_dir / "final.pt")
    (run_dir / "loss_curve.json").write_text(
        json.dumps({"hist": hist, "val_before": v0, "val_after": vl},
                   ensure_ascii=False), encoding="utf-8")
    log(f"SFT 完成 -> {run_dir/'final.pt'}")


if __name__ == "__main__":
    main()