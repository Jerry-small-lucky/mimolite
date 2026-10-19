# -*- coding: utf-8 -*-
"""
GRPO 代码版：可验证代码奖励驱动的策略优化（训练侧攻坚）
============================================================
与 grpo.py（GSM8K 数学版）的区别：
  1. 数据：模板合成代码题 synth_code_tpl.jsonl（600 条，无泄漏，断言即奖励）
  2. 奖励：子进程沙箱执行模型生成代码 + 断言 -> 通过 +1.0 / 无代码块 -0.5 / 失败 0.0
  3. prompt：与 SFT/评测一致（用户：{题面}\n + <|sep|> + 助手：），RL 学到的格式直接迁移评测
  4. 初始化：从当前最佳 sft_v8 出发，目标冲击 pass@1 与顽固区

用法（小规模验证）：
  python grpo_code.py --ckpt runs/sft_v8/final.pt --groups 24 --group-size 4 --max-new 400
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"

sys.path.insert(0, str(ROOT))
from eval_code import BAN_IMPORTS, extract_code, run_sandbox  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def compute_code_reward(completion: str, tests: str, timeout: int) -> tuple[float, dict]:
    """可验证代码奖励：通过断言 +1.0，无代码块 -0.5，其余 0.0。"""
    code = extract_code(completion)
    if code is None:
        return -0.5, {"pass": 0.0, "format": -0.5}
    ok, _ = run_sandbox(code, tests, timeout)
    if ok:
        return 1.0, {"pass": 1.0, "format": 0.0}
    return 0.0, {"pass": 0.0, "format": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(RUNS / "sft_v8/final.pt"))
    ap.add_argument("--data", type=str, default=str(DATA / "synth_code_tpl.jsonl"),
                    help="RL 训练数据（须为模型答不好的可验证代码题）")
    ap.add_argument("--out", type=str, default=str(RUNS / "grpo_code"))
    ap.add_argument("--groups", type=int, default=24, help="训练题目组数（每组 1 题）")
    ap.add_argument("--group-size", type=int, default=4, help="每题采样数 G")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.01)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--timeout", type=int, default=15, help="沙箱执行超时（秒）")
    ap.add_argument("--eval-every", type=int, default=6)
    ap.add_argument("--eval-n", type=int, default=16, help="评测取样题数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.groups = 2
        args.eval_every = 1
        args.group_size = 2

    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    if not Path(args.ckpt).exists():
        log(f"找不到 {args.ckpt}")
        return

    # ---- 训练数据：可验证代码题（默认模板合成；须是模型答不好的题才有 RL 信号） ----
    tpl = Path(args.data)
    if not tpl.exists():
        log(f"缺少 {tpl}")
        return
    rows = [json.loads(l) for l in tpl.open(encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    log(f"RL 训练题 {len(rows)} 条 <- {tpl.name}")

    # ---- 评测数据：评测集前 N 题（只评估，不训练） ----
    ev_items = [json.loads(l) for l in (DATA / "eval_code.jsonl").open(encoding="utf-8")]

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    bos, sep, eos, pad = meta["bos"], meta["sep"], meta["eos"], meta["pad"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备 {device} | 初始化自 {args.ckpt}")

    policy = MiMoLite(cfg).to(device)
    policy.load_state_dict(ck["model"], strict=False)
    ref = MiMoLite(cfg).to(device)
    ref.load_state_dict(ck["model"], strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    policy.train()

    def make_prompt_ids(q: str) -> list[int]:
        text = f"用户：{q}\n"
        return [bos] + tok.encode(text).ids + [sep] + tok.encode("助手：").ids

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # ---- 评测（低温采样，同 eval_code 协议） ----
    def evaluate() -> float:
        policy.eval()
        ok = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(device == "cuda")):
            for row in ev_items[:args.eval_n]:
                ids = torch.tensor([make_prompt_ids(row["question"])], device=device)
                out = policy.generate(ids, max_new_tokens=512, temperature=0.2,
                                      top_k=50, top_p=0.9, repetition_penalty=1.1,
                                      eos_id=eos, max_context=1024, generator=gen)
                comp = tok.decode(out[0][ids.shape[1]:].tolist())
                r, _ = compute_code_reward(comp, row["tests"], args.timeout)
                ok += 1 if r > 0.5 else 0
        policy.train()
        return ok / args.eval_n

    acc0 = evaluate()
    log(f"RL 前评测 {args.eval_n} 题 pass@1 = {acc0:.3f}")

    # ---- 训练循环 ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "grpo_log.jsonl").open("a", encoding="utf-8")
    hist = []
    t0 = time.time()
    cursor = 0

    def run_group(row) -> tuple[list[int], torch.Tensor, list[str], list[float]]:
        prompt = make_prompt_ids(row["question"])
        ids = torch.tensor([prompt] * args.group_size, device=device)
        was_training = policy.training
        policy.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(device == "cuda")):
            out = policy.generate(ids, max_new_tokens=args.max_new,
                                  temperature=args.temperature, top_k=50,
                                  top_p=args.top_p, eos_id=eos,
                                  max_context=1024, generator=gen)
        if was_training:
            policy.train()
        comps = [tok.decode(o[ids.shape[1]:].tolist()) for o in out]
        rews = [compute_code_reward(c, row["tests"], args.timeout)[0] for c in comps]
        return prompt, out, comps, rews

    def update(batch, lr_now: float):
        seqs, rewards, plens = [], [], []
        for prompt, out, comps, rews in batch:
            for o, r in zip(out, rews):
                seqs.append(o.tolist())
                rewards.append(r)
                plens.append(len(prompt))
        R = torch.tensor(rewards, device=device, dtype=torch.float32)
        if R.std() < 1e-6:
            return None, float(R.mean())
        adv = (R - R.mean()) / (R.std() + 1e-4)

        opt.zero_grad(set_to_none=True)
        n = len(seqs)
        losses = []
        for i, (seq, a, pl) in enumerate(zip(seqs, adv.tolist(), plens)):
            x = torch.tensor([seq[:-1]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = policy(x)["logits"].float()
                with torch.no_grad():
                    ref_logits = ref(x)["logits"].float()
            logp = F.log_softmax(logits, dim=-1)
            ref_logp = F.log_softmax(ref_logits, dim=-1)
            start = max(0, pl - 1)
            tgt = torch.tensor([seq[1:]], device=device)
            lp = logp[0, start:-1].gather(1, tgt[:, start:-1]).squeeze(1)
            rp = ref_logp[0, start:-1].gather(1, tgt[:, start:-1]).squeeze(1)
            kl = (rp - lp).mean()
            ratio = lp.exp()
            surr = -a * ratio
            loss = surr.mean() + args.kl_coef * kl
            (loss / n).backward()
            losses.append(float(loss))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()
        return float(np.mean(losses)), float(R.mean())

    for gi in range(args.groups):
        row = rows[cursor % len(rows)]
        cursor += 1
        batch = [run_group(row)]
        lr_now = args.lr * (0.5 * (1 + math.cos(math.pi * gi / max(1, args.groups))))
        loss, mean_r = update(batch, lr_now)

        acc = np.mean([1.0 if r > 0.5 else 0.0 for _, _, _, rews in batch for r in rews])
        rec = {"group": gi, "reward": mean_r, "acc": float(acc), "loss": loss,
               "lr": lr_now, "elapsed": time.time() - t0,
               "question": row["question"][:40]}
        hist.append(rec)
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()

        if (gi + 1) % 6 == 0 or loss is None:
            log(f"group {gi+1}/{args.groups} | 奖励 {mean_r:.3f} | 组内通过 {acc:.2f} | "
                f"loss {loss if loss is not None else float('nan'):.4f} | {(time.time()-t0)/60:.1f}m")
        if (gi + 1) % args.eval_every == 0 or gi + 1 == args.groups:
            a = evaluate()
            log(f"  >>> group {gi+1} 评测 pass@1 = {a:.3f} (RL 前 {acc0:.3f})")
            torch.save({"model": policy.state_dict(), "config": ck["config"],
                        "tokenizer_meta": meta, "step": gi + 1, "acc": a},
                       out_dir / "last.pt")

    torch.save({"model": policy.state_dict(), "config": ck["config"],
                "tokenizer_meta": meta, "step": args.groups}, out_dir / "final.pt")
    (out_dir / "history.json").write_text(
        json.dumps({"hist": hist, "acc_before": acc0}, ensure_ascii=False), encoding="utf-8")
    logf.close()
    log(f"GRPO(代码) 完成 -> {out_dir / 'final.pt'} | 总耗时 {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
