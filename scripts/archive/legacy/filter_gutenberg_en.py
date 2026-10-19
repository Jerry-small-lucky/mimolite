"""Gutenberg 英文语料语言过滤 —— 剔除非英语段（意大利语/法语/古语混杂）。
英文高频词占比 >= 3% 判为英语；否则丢弃。覆盖写回 hq_gutenberg_en.jsonl。
"""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "hq_gutenberg_en.jsonl"

STOP = set(
    "the and of to in that was his with he for at by from as not is it this her she had "
    "were you but have him which on so be all one them would their there when what more if "
    "no man out up will said they has an or we who been been".split()
)
WORD = re.compile(r"[a-z']+")
MIN_RATIO = 0.03


def main() -> None:
    total = kept = dropped = 0
    out = SRC.with_name("hq_gutenberg_en.filtered.jsonl")
    t0 = time.time()
    with SRC.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            words = WORD.findall(d["text"].lower())
            if not words:
                dropped += 1
                continue
            ratio = sum(1 for w in words if w in STOP) / len(words)
            if ratio < MIN_RATIO:
                dropped += 1
                continue
            fout.write(line)
            kept += 1
            total = kept + dropped
            if kept % 5000 == 0:
                print(f"{kept:,} kept / {dropped:,} dropped "
                      f"({kept/(time.time()-t0):.0f}/s)", flush=True)
    print(f"done: kept={kept:,} dropped={dropped:,} -> {out.name}")
    # 原子替换
    SRC.unlink()
    out.rename(SRC)


if __name__ == "__main__":
    main()
