# -*- coding: utf-8 -*-
"""
评测集隔离（手册第一步 · 路B 自建评测集）
====================================================================
1. 精确复现 sft.py 的验证集划分：build_example(Random(42).shuffle 前 60)
   —— 与训练/验证严格不重叠（直接复用 sft.build_example，保证管线一致）。
2. 从 data/synth_code.jsonl 剩余样本中按固定随机种子隔离 100 条写评测集。
3. 英文题目按 prepare_sft.py 同口径清洗 "Question: " 前缀（与训练分布一致）。

产出：
  data/eval_code.jsonl    {id, question, tests, lang, topic}（不含参考答案代码）
  logs/eval_set_meta.json 种子、验证集 id、评测集 id、被丢弃样本索引等审计信息

用法：python scripts/isolate_eval_set.py [--n 100] [--seed 2026]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sft import build_example  # noqa: E402  与 sft.py 训练数据管线完全一致

DATA = ROOT / "data"
LOGS = ROOT / "logs"
RAW = DATA / "synth_code.jsonl"
SFT_DATA = DATA / "synth_code_sft.jsonl"
OUT = DATA / "eval_code.jsonl"
META_OUT = LOGS / "eval_set_meta.json"

Q_PREFIX = re.compile(r"^Question\s*[:：]\s*", re.I)   # 与 prepare_sft.py 同口径


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    raw = [json.loads(l) for l in RAW.open(encoding="utf-8") if l.strip()]
    sft_lines = [json.loads(l) for l in SFT_DATA.open(encoding="utf-8") if l.strip()]
    assert len(raw) == len(sft_lines), f"raw {len(raw)} != sft {len(sft_lines)}，索引无法一一对应"

    # ---- 0. 核对 sft 文件与 raw 的顺序对应（含 EN Question: 清洗口径）----
    mismatch = 0
    for i in range(len(raw)):
        q = raw[i]["question"]
        if raw[i].get("lang") == "en" and Q_PREFIX.match(q):
            q = Q_PREFIX.sub("", q).strip()
        if sft_lines[i]["messages"][0]["content"].strip() != q.strip():
            mismatch += 1
            if mismatch <= 5:
                print(f"  [顺序不一致] idx {i}: raw='{q[:40]}' sft='{sft_lines[i]['messages'][0]['content'][:40]}'")
    if mismatch:
        print(f"[中止] sft 文件与 raw 顺序不一致 {mismatch} 处，无法按索引映射 id")
        sys.exit(1)
    print(f"顺序核对通过：{len(raw)} 条一一对应（含 EN 清洗口径）")

    # ---- 1. 复现 sft.py 验证集划分（Random(42).shuffle 后前 60）----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    MAX_LEN = 512   # sft.py 默认 --max-len

    idx_examples = []          # (原索引, example)
    dropped = []
    for i, r in enumerate(sft_lines):
        ex = build_example(tok, r["messages"], meta, MAX_LEN)
        if ex is None:
            dropped.append(i)
        else:
            idx_examples.append((i, ex))
    print(f"可用样本 {len(idx_examples)} / {len(sft_lines)}（丢弃 {len(dropped)} 条: {dropped}）")

    random.Random(42).shuffle(idx_examples)
    val_idx = [i for i, _ in idx_examples[:60]]          # sft.py val-n=60，行索引
    print(f"SFT 验证集 60 条行索引: {sorted(val_idx)[:10]} ...")

    # ---- 2. 从剩余样本（按行索引排除）隔离评测集 ----
    # 注意：synth_code.jsonl 的 id 字段因历史两次运行区间重叠而重复（610 题唯一，364 id 唯一），
    # 因此以【行索引】为稳定标识，与 sft.py 的洗牌粒度一致。
    val_set = set(val_idx)
    pool = [(i, d) for i, d in enumerate(raw) if i not in val_set]
    assert len(pool) == len(raw) - len(val_idx)
    qs = [d["question"] for _, d in pool]
    assert len(qs) == len(set(qs)), "剩余样本存在重复题目，隔离失效"
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    evals = pool[: args.n]
    print(f"评测集 {len(evals)} 条（seed={args.seed}），与 SFT 训练/验证零重叠")

    # ---- 3. 写评测集（题目按训练口径清洗；不含参考答案代码）----
    # 评测集 id 重新编号保证唯一；src_id/src_line 保留溯源
    with OUT.open("w", encoding="utf-8") as f:
        for k, (src_line, d) in enumerate(evals):
            q = d["question"]
            if d.get("lang") == "en" and Q_PREFIX.match(q):
                q = Q_PREFIX.sub("", q).strip()
            f.write(json.dumps({
                "id": f"eval_{k:05d}", "src_id": d["id"], "src_line": src_line,
                "question": q, "tests": d["tests"],
                "lang": d.get("lang"), "topic": d.get("topic"),
            }, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "n_eval": len(evals), "seed_eval": args.seed, "seed_val": 42,
        "n_val": len(val_idx), "val_indices": sorted(val_idx),
        "eval_src_lines": [i for i, _ in evals],
        "eval_ids": [d["id"] for _, d in evals],
        "dropped_by_sft": dropped,
        "lang_counts": {k: sum(1 for _, d in evals if d.get("lang") == k) for k in ("zh", "en")},
        "note": "val 集复现自 sft.py Random(42).shuffle 前 60（按行索引）；评测集从剩余样本按 seed 抽取，与训练/验证严格不重叠。"
                "synth_code.jsonl 的 id 字段因历史两次运行重叠而重复（610 题唯一），故以行索引为稳定标识。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成：{OUT}（{len(evals)} 条），元信息 {META_OUT}")


if __name__ == "__main__":
    main()
