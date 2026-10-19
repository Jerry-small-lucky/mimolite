"""
A-001 专用高质量子池构建
=========================
从清洗后语料按 token 预算抽样，叠加 Gutenberg 与合成数据（过采样），
产出 a001_zh.jsonl / a001_en.jsonl 两个文件（{"text","lang"} 格式），
供 A-001 三配置训练使用（pretrain.py --zh-data / --en-data）。

配比设计（A-001 预算 59M token，池子略大于训练量）：
  zh 总预算 ~48M：清洗抽样 ~40M + 合成全量 ~8M
  en 总预算 ~24M：清洗抽样 ~12M + Gutenberg 全量 ~5M + 合成全量 ~7M
合成+Gutenberg 合计占池 ~28%，远高于自然占比（<1%），让高质量数据真正参与训练。
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOG / "build_a001_pool.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def est_tokens(text: str, lang: str) -> int:
    return max(1, int(len(text) / (1.85 if lang == "zh" else 2.33)))


def copy_segments(src_name: str, out_f, lang: str, budget_tok: int | None = None,
                  shuffle: bool = False, seed: int = 42) -> tuple[int, int]:
    """拷贝段落到输出；budget_tok 限制 token 总量；返回 (段数, token 数)。"""
    src = DATA / src_name
    if not src.exists():
        log(f"跳过 {src_name}：不存在")
        return 0, 0
    segs = [json.loads(l)["text"] for l in src.open(encoding="utf-8")]
    if shuffle:
        random.seed(seed)
        random.shuffle(segs)
    n = tok = 0
    for s in segs:
        t = est_tokens(s, lang)
        if budget_tok is not None and tok + t > budget_tok:
            continue
        out_f.write(json.dumps({"text": s, "lang": lang}, ensure_ascii=False) + "\n")
        n += 1
        tok += t
    return n, tok


def main() -> None:
    zh_budget = 48_000_000
    en_budget = 24_000_000
    zh_clean_budget = zh_budget - 8_000_000   # 给合成留 8M
    en_clean_budget = en_budget - 12_000_000  # 给 gutenberg+合成留 12M

    with (DATA / "a001_zh.jsonl").open("w", encoding="utf-8") as zf, \
         (DATA / "a001_en.jsonl").open("w", encoding="utf-8") as ef:
        log("--- zh ---")
        n, tok = copy_segments("clean_zh.jsonl", zf, "zh", zh_clean_budget, shuffle=True)
        log(f"clean_zh: {n:,} 段 {tok/1e6:.1f}M tok")
        n2, tok2 = copy_segments("hq_synth_zh.jsonl", zf, "zh", None)
        log(f"synth_zh: {n2:,} 段 {tok2/1e6:.1f}M tok")
        zh_total = tok + tok2
        log("--- en ---")
        n, tok = copy_segments("clean_en.jsonl", ef, "en", en_clean_budget, shuffle=True)
        log(f"clean_en: {n:,} 段 {tok/1e6:.1f}M tok")
        n2, tok2 = copy_segments("hq_gutenberg_en.jsonl", ef, "en", None)
        log(f"gutenberg_en: {n2:,} 段 {tok2/1e6:.1f}M tok")
        n3, tok3 = copy_segments("hq_synth_en.jsonl", ef, "en", None)
        log(f"synth_en: {n3:,} 段 {tok3/1e6:.1f}M tok")
        en_total = tok + tok2 + tok3

    log(f"完成：a001_zh {zh_total/1e6:.1f}M tok, a001_en {en_total/1e6:.1f}M tok, "
        f"合计 {(zh_total+en_total)/1e6:.1f}M tok")


if __name__ == "__main__":
    main()
