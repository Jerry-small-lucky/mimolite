"""
阶段二数据准备：指令微调语料
==============================

数据源（实测 hf-mirror 可用）：
  - silk-road/alpaca-data-gpt4-chinese   中文指令数据（GPT-4 生成）
  - open-thoughts/OpenThoughts-114k      长链思维链推理数据
  - AI-MO/NuminaMath-CoT                 数学推理 CoT
  - Jerry 自建语料                        F:\AI\datasets\my_corpus\train.jsonl

统一输出 data/sft.jsonl，字段 {messages:[{role,content}...]}，供 sft.py 使用。

用法：
  python prepare_sft_data.py --n-cn 60000 --n-thought 40000 --n-math 20000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "sft.jsonl"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def clean(t: str, limit: int = 1200) -> str:
    t = (t or "").strip()
    return t[:limit]


def from_alpaca_cn(n: int, out) -> int:
    from datasets import load_dataset
    log(f"拉取中文指令数据 alpaca-data-gpt4-chinese，目标 {n:,}")
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    c = 0
    for r in ds:
        q = clean(r.get("instruction_zh", ""))
        inp = clean(r.get("input_zh", ""), 600)
        a = clean(r.get("output_zh", ""))
        if not q or not a:
            continue
        content = f"{q}\n\n{inp}" if inp else q
        out.write(json.dumps({"messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": a}], "src": "alpaca-cn"},
            ensure_ascii=False) + "\n")
        c += 1
        if c >= n:
            break
    log(f"  -> {c:,} 条")
    return c


def from_openthoughts(n: int, out) -> int:
    from datasets import load_dataset
    log(f"拉取思维链数据 OpenThoughts，目标 {n:,}")
    try:
        ds = load_dataset("open-thoughts/OpenThoughts-114k", "default", split="train")
    except Exception as e:
        log(f"  跳过：{type(e).__name__} {str(e)[:120]}")
        return 0
    c = 0
    for r in ds:
        convs = r.get("conversations") or []
        if len(convs) < 2:
            continue
        u = clean(convs[0].get("value"), 600)
        a = clean(convs[1].get("value"), 1600)
        if not u or not a:
            continue
        out.write(json.dumps({"messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a}], "src": "openthoughts"},
            ensure_ascii=False) + "\n")
        c += 1
        if c >= n:
            break
    log(f"  -> {c:,} 条")
    return c


def from_numina(n: int, out) -> int:
    from datasets import load_dataset
    log(f"拉取数学推理数据 NuminaMath，目标 {n:,}")
    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        log(f"  跳过：{type(e).__name__} {str(e)[:120]}")
        return 0
    c = 0
    for r in ds:
        msgs = r.get("messages")
        if not msgs or len(msgs) < 2:
            continue
        u = clean(msgs[0].get("content"), 600)
        a = clean(msgs[1].get("content"), 1600)
        if not u or not a:
            continue
        out.write(json.dumps({"messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a}], "src": "numina"},
            ensure_ascii=False) + "\n")
        c += 1
        if c >= n:
            break
    log(f"  -> {c:,} 条")
    return c


def from_local(out) -> int:
    """Jerry 自己那份翻译语料，全量带上。"""
    p = Path(r"F:\AI\datasets\my_corpus\train.jsonl")
    if not p.exists():
        log("本地语料不存在，跳过")
        return 0
    c = 0
    with p.open(encoding="utf-8") as f, out.open("a", encoding="utf-8") as o:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = r.get("instruction", "")
            inp = r.get("input", "")
            a = r.get("output", "")
            if not (q and a):
                continue
            o.write(json.dumps({"messages": [
                {"role": "user", "content": f"{q}\n\n{inp}" if inp else q},
                {"role": "assistant", "content": a}], "src": "local"},
                ensure_ascii=False) + "\n")
            c += 1
    log(f"  本地语料 -> {c} 条")
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cn", type=int, default=60_000)
    ap.add_argument("--n-thought", type=int, default=30_000)
    ap.add_argument("--n-math", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        from_alpaca_cn(args.n_cn, out)
        from_openthoughts(args.n_thought, out)
        from_numina(args.n_math, out)
    total = sum(1 for _ in tmp.open(encoding="utf-8"))
    from_local(tmp)
    total = sum(1 for _ in tmp.open(encoding="utf-8"))

    tmp.replace(OUT)
    log(f"SFT 数据集完成：{total:,} 条 -> {OUT}")

    from collections import Counter
    cnt = Counter(json.loads(l).get("src", "?") for l in OUT.open(encoding="utf-8"))
    for k, v in cnt.most_common():
        log(f"  {k}: {v:,}")