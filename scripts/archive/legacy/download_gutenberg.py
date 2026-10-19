"""
精选 Gutenberg 经典书目下载 —— 英文高质量语料补充
==================================================
串行单线程下载（项目纪律：下载永远单进程单线程串行），失败重试 2 次后跳过。
处理：去 Gutenberg 样板头尾 -> 清脚注/下划线 -> 切段 1200 -> 质量过滤。
输出：data/hq_gutenberg_en.jsonl（增量写盘，进度日志 logs/gutenberg.log）
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根
DATA = ROOT / "data"
LOG = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)
OUT = DATA / "hq_gutenberg_en.jsonl"

# 经典公共领域书目（pg 编号）；404 的会自动跳过
BOOK_IDS = [
    11, 46, 74, 84, 98, 100, 120, 174, 219, 408, 521, 582, 768, 842, 996,
    1000, 1260, 1342, 1400, 1661, 1727, 1952, 1998, 2148, 2441, 2554, 2591,
    2600, 2701, 2800, 2852, 3207, 3329, 3460, 35840, 4260, 4300, 5000, 6130,
]

START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\Z", re.I | re.S)
LATIN = re.compile(r"[A-Za-z]")
SPACE_CLEAN = re.compile(r"[ \t]+")
BLANK_NL = re.compile(r"\n{3,}")
FOOT_RE = re.compile(r"(?m)^\s*\[\w\s*\]\s*$")  # 脚注占位行
UNDER_RE = re.compile(r"(?<=\w)_(?=\w)")  # 词内下划线


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOG / "gutenberg.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def quality(text: str) -> float:
    return (len(LATIN.findall(text)) + text.count(" ")) / max(1, len(text))


def clean(text: str) -> str:
    t = UNDER_RE.sub(" ", text)
    t = FOOT_RE.sub(" ", t)
    t = SPACE_CLEAN.sub(" ", t)
    t = BLANK_NL.sub("\n\n", t)
    return t.strip()


def split_segments(text: str, seg_chars: int = 1200) -> list[str]:
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


def fetch(pg_id: int) -> str | None:
    url = f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            log(f"  pg{pg_id} 尝试 {attempt+1} 失败: {e}")
            time.sleep(5)
    return None


def main() -> None:
    done = 0
    total_seg = 0
    with OUT.open("a", encoding="utf-8") as fout:
        for pg_id in BOOK_IDS:
            raw = fetch(pg_id)
            if raw is None:
                log(f"pg{pg_id} 下载失败，跳过")
                continue
            m = START_RE.search(raw)
            body = raw[m.end():] if m else raw
            m2 = END_RE.search(body)
            if m2:
                body = body[: m2.start()]
            t = clean(body)
            segs = split_segments(t)
            kept = 0
            for s in segs:
                if len(s) < 150 or quality(s) < 0.60:
                    continue
                fout.write(json.dumps({"text": s, "lang": "en"}, ensure_ascii=False) + "\n")
                kept += 1
            done += 1
            total_seg += kept
            log(f"pg{pg_id}: {len(body):,} 字符 -> {kept:,} 段（累计 {done}/{len(BOOK_IDS)} 本）")
    log(f"Gutenberg 完成：{done} 本，共 {total_seg:,} 段 -> {OUT.name}")


if __name__ == "__main__":
    main()
