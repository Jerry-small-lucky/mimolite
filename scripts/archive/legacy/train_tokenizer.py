"""
训练自有 BPE tokenizer（中英混合）
=================================

选型说明：
  采用 **byte-level BPE**（GPT-2 / Qwen 同一路线），原因：
  - 中文按字节合并，天然无 OOV，词表不需要为生僻字留位
  - llama.cpp 对该格式原生支持，导出 GGUF 后直接可用
  - 中英混排时英文仍有高压缩率（常见词合并成单 token）

小米 MiMo 的 tokenizer 同样是自研的；本机规模下自训 16K 词表，
约 108M 参数模型中 embedding 占 12.6M（11.7%），比例合理。

用法：
  python train_tokenizer.py --vocab 16384 --sample 300000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "tokenizer"
OUT.mkdir(exist_ok=True)

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|sep|>"]
PAD, BOS, EOS, SEP = 0, 1, 2, 3


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def corpus_iter(paths: list[Path], limit: int, max_chars: int = 1500):
    """
    流式产出训练文本。

    注意：byte-level 预分词会把中文切成单字，喂给 BPE 训练器的词频表会
    随输入量线性膨胀（20 万行中文 ≈ 20GB 内存）。因此这里做两件事：
      1. 每行截断到 max_chars（切段时本就是 <=1200 字符）
      2. 默认样本量控制在 8 万行左右 —— 对 16K 词表完全够用
    """
    n = 0
    for p in paths:
        if not p.exists():
            log(f"跳过不存在的 {p.name}")
            continue
        log(f"读取 {p.name}")
        with p.open(encoding="utf-8") as f:
            for line in f:
                if n >= limit:
                    return
                try:
                    text = json.loads(line)["text"][:max_chars]
                except Exception:
                    continue
                if text.strip():
                    yield text
                    n += 1
                if n % 20000 == 0:
                    log(f"  已扫描 {n:,} 段")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--sample", type=int, default=80_000)
    ap.add_argument("--zh-share", type=float, default=0.7, help="中文样本占比")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    # byte-level BPE
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        min_frequency=args.min_freq,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    log(f"开始训练 tokenizer，vocab={args.vocab}, 样本 {args.sample:,} 段")
    t0 = time.time()
    n_zh = int(args.sample * args.zh_share)
    n_en = args.sample - n_zh
    # 注意：train_from_iterator 只能调用一次（多次调用会重置模型），
    # 因此把中英文两个来源串成同一条流。
    def mixed():
        yield from corpus_iter([DATA / "pretrain_zh.jsonl"], n_zh, args.max_chars)
        yield from corpus_iter([DATA / "pretrain_en.jsonl"], n_en, args.max_chars)

    tok.train_from_iterator(mixed(), trainer=trainer)
    log(f"训练完成，耗时 {time.time()-t0:.1f}s，实际词表 {tok.get_vocab_size()}")

    path = OUT / "tokenizer.json"
    tok.save(str(path))
    log(f"已保存 -> {path}")

    # ---- 自检 ----
    log("--- 编码自检 ---")
    for s in [
        "数学是对数量、结构、空间变化以及集合等抽象对象的知识 difficulties。",
        "Deep learning is fascinating, and 深度学习很有趣。",
        "深度学习很有趣，Deep learning is fascinating!",
        "def train(model, batch):\n    return model(batch)",
    ]:
        ids = tok.encode(s).ids
        back = tok.decode(ids)
        ok = "OK " if back.strip() == s.strip() else "DIFF"
        log(f"  [{ok}] {len(s):3d}字 -> {len(ids):3d}tok (压缩率 {len(s)/max(len(ids),1):.2f}) {s[:40]}")

    # 中文压缩率统计
    zh = "人工智能与机器学习是当前科技发展的重要方向之一。"
    ids = tok.encode(zh).ids
    log(f"  中文压缩率：{len(zh)}字 -> {len(ids)}tok ({len(zh)/len(ids):.2f} 字/token)")

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "pad": PAD, "bos": BOS, "eos": EOS, "sep": SEP,
        "type": "byte-level BPE",
    }
    (OUT / "tokenizer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"元信息 -> {OUT/'tokenizer_meta.json'}")


if __name__ == "__main__":
    main()