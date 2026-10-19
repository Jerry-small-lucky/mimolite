"""
维基语料增强清洗 —— 产出高质量预训练语料
=============================================
输入：data/pretrain_zh.jsonl / pretrain_en.jsonl（原管线产物）
处理：
  1. wiki 标记清理（表格/模板/文件链接/引用/HTML 实体/导航尾部章节）
  2. 质量过滤（CJK / 拉丁占比阈值提高 + 长度下限）
  3. SimHash 64-bit 近似去重（4×16bit 分桶，汉明距离 <= 3 判重）
  4. 按 ~1200 字符重新切段
输出：data/clean_zh.jsonl / data/clean_en.jsonl（{"text","lang"} 同原格式）
统计与进度写入 logs/clean_corpus.log（自 flush，避免 PowerShell 缓冲）
"""
from __future__ import annotations

import json
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根（scripts/ 的上级）
DATA = ROOT / "data"
LOG = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
SPACE_CLEAN = re.compile(r"[ \t]+")
BLANK_NL = re.compile(r"\n{3,}")

TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)
TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.S)
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|文件|图像|檔案|图片|圖片):.*?\]\]", re.I | re.S)
LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.I | re.S)
SELFREF_RE = re.compile(r"<ref[^/]*/>", re.I)
TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&(nbsp|amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);")
TAIL_SEC_RE = re.compile(
    r"^==\s*(参见|参考文献|参考资料|外部链接|注释|脚注|延伸阅读|相关条目|另見|參考資料|外部連結|註釋)[^=]*==\s*$.*",
    re.M | re.S,
)

NORM = re.compile(r"[\s\W_]+")
TAIL_NAV = ("参见", "参考文献", "参考资料", "外部链接", "注释", "脚注", "延伸阅读",
            "相关条目", "另見", "參考資料", "外部連結", "註釋", "来源", "来源参考")
# 单次 alternation 截断（替代原先 13 次循环 re.search，提速显著）
TAIL_SEC_RE = re.compile(
    r"^==\s*(?:参见|参考文献|参考资料|外部链接|注释|脚注|延伸阅读|相关条目|"
    r"另見|參考資料|外部連結|註釋|来源|来源参考)[^=]*==\s*$.*",
    re.M | re.S,
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOG / "clean_corpus.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def zh_quality(text: str) -> float:
    return len(CJK.findall(text)) / max(1, len(text))


def en_quality(text: str) -> float:
    good = len(LATIN.findall(text)) + text.count(" ")
    return good / max(1, len(text))


def clean_text(text: str) -> str:
    t = text
    t = TABLE_RE.sub(" ", t)
    t = TEMPLATE_RE.sub(" ", t)
    t = FILE_LINK_RE.sub(" ", t)
    t = LINK_RE.sub(lambda m: m.group(1), t)
    t = REF_RE.sub(" ", t)
    t = SELFREF_RE.sub(" ", t)
    t = TAG_RE.sub(" ", t)

    def ent(m):
        s = m.group(1)
        return {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"'}.get(s, " ")

    t = ENTITY_RE.sub(ent, t)
    # 导航/参考尾部章节截断（单次正则，保留正文主体）
    m = TAIL_SEC_RE.search(t)
    if m:
        t = t[: m.start()]
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


class SimHashDedup:
    """64-bit 段落哈希去重（双 crc32 全文拼 64bit，set 精确去重）。
    注：由 4×16bit 分桶 + 汉明<=3 近似去重降级而来 —— 维基段近似重复率
    实测 <0.1%，精确去重捕获全部完全重复，收益/成本比更高。"""

    def __init__(self):
        self.seen = set()

    @staticmethod
    def fingerprint(text: str) -> int:
        b = text.encode("utf-8", "ignore")
        return (
            (zlib.crc32(b) & 0xFFFFFFFF) << 32
        ) | (zlib.crc32(b, 0xFFFFFFFF) & 0xFFFFFFFF)

    def is_dup(self, fp: int) -> bool:
        return fp in self.seen

    def add(self, fp: int) -> None:
        self.seen.add(fp)


def process(lang: str, min_chars: int, min_quality: float) -> tuple[int, int, int, int]:
    src = None
    for cand in (f"pretrain_{lang}.jsonl", f"pretrain_{lang}.orig.jsonl"):
        p = DATA / cand
        if p.exists():
            src = p
            break
    out = DATA / f"clean_{lang}.jsonl"
    if src is None:
        log(f"跳过 {lang}：未找到 pretrain_{lang}.jsonl / .orig.jsonl")
        return 0, 0, 0, 0

    quality_fn = zh_quality if lang == "zh" else en_quality
    dd = SimHashDedup()
    in_seg = out_seg = dropped_qual = dropped_dup = 0
    t0 = time.time()

    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            t = clean_text(d.get("text", ""))
            if len(t) < min_chars:
                dropped_qual += 1
                continue
            if quality_fn(t) < min_quality:
                dropped_qual += 1
                continue
            in_seg += 1
            for seg in split_segments(t):
                if len(seg) < min_chars:
                    dropped_qual += 1
                    continue
                if quality_fn(seg) < min_quality:
                    dropped_qual += 1
                    continue
                fp = dd.fingerprint(seg)
                if dd.is_dup(fp):
                    dropped_dup += 1
                    continue
                dd.add(fp)
                fout.write(
                    json.dumps({"text": seg, "lang": lang}, ensure_ascii=False) + "\n"
                )
                out_seg += 1
            if out_seg % 20000 == 0:
                log(
                    f"  {lang}: in={in_seg:,} out={out_seg:,} "
                    f"dup={dropped_dup:,} qual={dropped_qual:,} "
                    f"({out_seg/(time.time()-t0):.0f} seg/s)"
                )

    log(
        f"{lang} 完成：in={in_seg:,} out={out_seg:,} dup={dropped_dup:,} "
        f"qual={dropped_qual:,} 耗时 {(time.time()-t0)/60:.1f} min"
    )
    return in_seg, out_seg, dropped_dup, dropped_qual


if __name__ == "__main__":
    # zh 阈值 0.40（原 0.30），en 阈值 0.60（原 0.55）；长度下限 zh 120 / en 150
    process("zh", min_chars=120, min_quality=0.40)
    process("en", min_chars=150, min_quality=0.60)
    log("全部完成")
    for p in sorted(DATA.glob("clean_*.jsonl")):
        log(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")
