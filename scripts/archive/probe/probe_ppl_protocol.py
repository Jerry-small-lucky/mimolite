"""
perplexity 协议对照实验：定位 0.22 nats 差距的真正来源
=========================================================

前面的排查已经把所有「模型侧」的可能性都排除了：
  - GGUF 张量与 checkpoint 逐位相同（max|diff| = 0）
  - hparams 与官方 Qwen3-1.7B 逐项一致（rope type=2、rope scaling=linear 是 qwen3 的默认值）
  - n_swa = 0，llama.cpp 没有误用滑动窗口
  - tokenizer 分词逐 id 相同（修复 pre=gpt-2 之后）
  - fp32 / bf16 / fp16 在 PyTorch 侧 loss 完全一致

那么剩下的唯一解释是：**llama.cpp 的 llama-perplexity 与我们手写的评估循环
协议不同**（切块方式、每块是否重加 BOS、是否跳过块首 token、是否有滑窗重叠）。

做法：用小规模（300 token、-c 100）让 llama.cpp 切成 3 块，
然后在 PyTorch 侧枚举各种切分协议，看哪一种能对上 llama.cpp 的数字。

用法：python probe_ppl_protocol.py
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_text(n_tokens: int, tok) -> str:
    """拼一段恰好约 n_tokens 的中文文本。"""
    parts, n = [], 0
    with (DATA / "pretrain_zh.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 8000:
                continue
            t = json.loads(line)["text"][:600]
            parts.append(t)
            n += len(tok.encode(t).ids)
            if n >= n_tokens:
                break
    return "\n".join(parts)


@torch.no_grad()
def chunked_loss(model, ids, chunk, add_bos_per_chunk, skip_first, device):
    """
    在 PyTorch 侧复现「切块 + 每块独立前向」的 loss。
    ids 已含开头 BOS。
    """
    tot, n = 0.0, 0
    bos = ids[0]
    body = ids[1:]                      # 去掉全局 BOS
    for s in range(0, len(body), chunk):
        piece = body[s:s + chunk]
        if len(piece) < 2:
            break
        if add_bos_per_chunk:
            piece = [bos] + piece
        x = torch.tensor([piece], device=device)
        if skip_first:
            x = x[:, :-1]
        logits = model(x)["logits"].float()
        tgt = x[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tgt.reshape(-1), reduction='mean')
        k = int(tgt.numel())
        tot += loss.item() * k
        n += k
    return tot / n


@torch.no_grad()
def full_loss(model, ids, device):
    """无切块，整段一次前向（上下文完整，loss 的下界）。"""
    x = torch.tensor([ids], device=device)
    logits = model(x)["logits"].float()
    tgt = x[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tgt.reshape(-1), reduction='mean')
    return loss.item()


def main() -> None:
    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    ckpt = str(ROOT / "runs/pretrain/snap500.pt")
    gguf = str(ROOT / "out/mimolite-500-v2.gguf")
    CHUNK = 100

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    bos = meta["bos"]

    text = get_text(300, tok)
    ids = [bos] + tok.encode(text).ids
    log(f"测试文本 {len(text)} 字 -> {len(ids)} token（含 BOS），切块粒度 {CHUNK}")

    # ---- PyTorch 侧 ----
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = MiMoLite(cfg).to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    for l in m.layers:            # 强制全注意力，对应 --attn-mode full 导出
        l.attn.is_global = True

    variants = {
        "整段一次前向（无切块）": full_loss(m, ids, device),
        f"切块{len(ids)//CHUNK}块·块首不加BOS": chunked_loss(m, ids, CHUNK, False, False, device),
        f"切块·每块重加BOS": chunked_loss(m, ids, CHUNK, True, False, device),
        f"切块·跳过块首token": chunked_loss(m, ids, CHUNK, False, True, device),
        f"切块·加BOS且跳块首": chunked_loss(m, ids, CHUNK, True, True, device),
    }

    # ---- llama.cpp 侧 ----
    LOGS.mkdir(exist_ok=True)
    (LOGS / "_proto.txt").write_text(text, encoding="utf-8")
    out = LOGS / "_proto_llama.txt"
    cmd = [r"F:\AI\tools\llama.cpp\llama-perplexity.exe",
           "-m", gguf, "-f", "_proto.txt", "-c", str(CHUNK), "-b", str(CHUNK)]
    with out.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, cwd=str(LOGS), timeout=900)
    raw = out.read_text(encoding="utf-8", errors="replace")
    (LOGS / "proto_raw.txt").write_text(raw, encoding="utf-8")

    ppls = []
    for line in raw.splitlines():
        if "Final estimate" in line:
            for s in line.replace("=", " ").replace(",", " ").split():
                try:
                    ppls.append(float(s))
                    break
                except ValueError:
                    pass
    # 也抓逐块数值
    chunks = []
    for line in raw.splitlines():
        if line.startswith("[") and "]" in line and "ppl" not in line.lower():
            for part in line.strip("[],").split(","):
                try:
                    chunks.append(float(part))
                except ValueError:
                    pass

    print()
    print("=" * 74)
    print(f"{'协议 / 实现':<34}{'loss':>10}{'ppl':>12}")
    print("-" * 74)
    for k, v in variants.items():
        print(f"{k:<34}{v:>10.4f}{np.exp(v):>12.2f}")
    if ppls:
        ll = float(np.log(ppls[0]))
        print(f"{'llama.cpp (final estimate)':<34}{ll:>10.4f}{ppls[0]:>12.2f}")
    print("-" * 74)
    if chunks:
        print(f"llama.cpp 逐块 ppl: {chunks}")
        print(f"  -> 对应 loss: {[round(float(np.log(c)), 4) for c in chunks]}")
    print("=" * 74)

    if ppls:
        ll = float(np.log(ppls[0]))
        print()
        print("与 llama.cpp 最接近的 PyTorch 协议：")
        best = min(variants.items(), key=lambda kv: abs(kv[1] - ll))
        for k, v in variants.items():
            mark = "  <== 最接近" if k == best[0] else ""
            print(f"  {k:<34}{v:>10.4f}  差 {v-ll:+.4f}{mark}")
        print()
        if abs(best[1] - ll) < 0.05:
            print("结论：差距来自**评估协议差异**，不是模型不一致。")
            print("      GGUF 导出是忠实的，之前的对比方式本身就不对等。")
        else:
            print("结论：仍有无法用协议差异解释的缺口，需继续排查。")
    print()
    print("原始 llama 输出见 logs/proto_raw.txt")


if __name__ == "__main__":
    main()