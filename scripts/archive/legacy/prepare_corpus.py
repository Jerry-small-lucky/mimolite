"""
语料准备 —— 从维基百科流式抓取中英文语料，清洗后落盘 jsonl
=============================================================

数据源（均通过 hf-mirror 实测可用）：
  - wikimedia/wikipedia / 20231101.zh   中文维基百科（Parquet 分片，流式友好）
  - wikimedia/wikipedia / 20231101.en   英文维基百科
  - openai/gsm8k                        后续 GRPO 推理训练用

清洗规则（对应小米 MiMo 技术报告中的数据管线思路）：
  1. 长度过滤：剔除过短片段与超长噪声
  2. 字符质量：中文占比过滤掉导航/列表/乱码页
  3. 去重：对归一化文本前缀做哈希去重
  4. 分段：按 ~1200 字符切段，适配 seq_len=512 的高效打包

用法：
  python prepare_corpus.py --zh 800000 --en 400000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
SPACE_CLEAN = re.compile(r"[ \t]+")
BLANK_NL = re.compile(r"\n{3,}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def zh_quality(text: str) -> float:
    """中文语料质量分：CJK 字符占比。"""
    if not text:
        return 0.0
    return len(CJK.findall(text)) / len(text)


def en_quality(text: str) -> float:
    """英文语料质量分：拉丁字母 + 空格占比。"""
    if not text:
        return 0.0
    good = len(LATIN.findall(text)) + text.count(" ")
    return good / len(text)


def clean(text: str) -> str:
    text = SPACE_CLEAN.sub(" ", text)
    text = BLANK_NL.sub("\n\n", text)
    return text.strip()


def split_segments(text: str, seg_chars: int = 1200) -> list[str]:
    """按段落边界切段，尽量不切断句子。"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    segs, cur, size = [], [], 0
    for p in paras:
        if size + len(p) > seg_chars and cur:
            segs.append("\n".join(cur))
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        segs.append("\n".join(cur))
    return segs


def harvest(
    repo: str,
    config: str,
    text_key: str,
    title_key: str | None,
    target_docs: int,
    out_path: Path,
    lang: str,
    min_chars: int,
    max_chars: int,
    min_quality: float,
    seg_chars: int,
    max_segments: int,
) -> int:
    """流式抓取并写出 jsonl，返回写出的段数。"""
    from datasets import load_dataset

    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"{out_path.name} 已存在，跳过（删除文件可重跑）")
        return sum(1 for _ in out_path.open(encoding="utf-8"))

    log(f"开始抓取 {lang}: {repo}/{config}，目标 {target_docs:,} 篇 -> {out_path.name}")
    ds = load_dataset(repo, config, split="train", streaming=True)

    quality_fn = zh_quality if lang == "zh" else en_quality
    seen: set[str] = set()
    kept_docs = 0
    seg_count = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for row in ds:
            text = row.get(text_key) or ""
            if not (min_chars <= len(text) <= max_chars):
                continue
            if quality_fn(text) < min_quality:
                continue

            key = hashlib.md5(text[:120].encode("utf-8", "ignore")).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            title = row.get(title_key) if title_key else None
            segs = split_segments(clean(text), seg_chars)
            if not segs:
                continue

            fout.write(json.dumps({
                "text": (f"{title}\n" + segs[0]) if title else segs[0],
                "lang": lang,
            }, ensure_ascii=False) + "\n")
            seg_count += 1
            for s in segs[1:]:
                if len(s) < 100:
                    continue
                fout.write(json.dumps({"text": s, "lang": lang}, ensure_ascii=False) + "\n")
                seg_count += 1
            kept_docs += 1

            if kept_docs % 50000 == 0:
                el = time.time() - t0
                log(f"  {lang}: {kept_docs:,} 篇 / {seg_count:,} 段 "
                    f"({kept_docs/max(el,1):.0f} 篇/s, 已用 {el/60:.1f} 分钟)")
            if kept_docs >= target_docs or seg_count >= max_segments:
                break

    log(f"{lang} 完成：{kept_docs:,} 篇 -> {seg_count:,} 段，"
        f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    return seg_count


def fetch_gsm8k() -> None:
    """抓取 GSM8K，供后续 GRPO 可验证奖励训练使用。"""
    out = DATA / "gsm8k.jsonl"
    if out.exists() and out.stat().st_size > 0:
        log("gsm8k.jsonl 已存在，跳过")
        return
    from datasets import load_dataset
    log("抓取 GSM8K（GRPO 训练用）")
    ds = load_dataset("openai/gsm8k", "main")
    with out.open("w", encoding="utf-8") as f:
        for split in ("train", "test"):
            for r in ds[split]:
                ans = r["answer"]
                final = ans.split("####")[-1].strip() if "####" in ans else ""
                f.write(json.dumps({
                    "question": r["question"],
                    "solution": ans,
                    "answer": final,
                    "split": split,
                }, ensure_ascii=False) + "\n")
    log(f"gsm8k 完成：{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", type=int, default=800_000, help="中文目标篇数")
    ap.add_argument("--en", type=int, default=400_000, help="英文目标篇数")
    ap.add_argument("--seg-chars", type=int, default=1200)
    ap.add_argument("--max-segments", type=int, default=1_200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-en", action="store_true")
    ap.add_argument("--skip-zh", action="store_true")
    ap.add_argument("--gsm8k-only", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    if args.gsm8k_only:
        fetch_gsm8k()
        sys.exit(0)

    if not args.skip_zh:
        harvest(
            "wikimedia/wikipedia", "20231101.zh", "text", "title",
            args.zh, DATA / "pretrain_zh.jsonl", "zh",
            min_chars=150, max_chars=40000, min_quality=0.30,
            seg_chars=args.seg_chars, max_segments=args.max_segments,
        )
    if not args.skip_en:
        harvest(
            "wikimedia/wikipedia", "20231101.en", "text", "title",
            args.en, DATA / "pretrain_en.jsonl", "en",
            min_chars=300, max_chars=60000, min_quality=0.55,
            seg_chars=args.seg_chars, max_segments=args.max_segments,
        )
    fetch_gsm8k()

    log("全部完成：")
    for p in sorted(DATA.glob("*.jsonl")):
        log(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")