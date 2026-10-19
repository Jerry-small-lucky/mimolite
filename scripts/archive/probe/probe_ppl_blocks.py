"""
逐块严格对照：确认 0.22 nats 是否只是「比较对象不对等」
=========================================================

关键发现（见 logs/proto_raw.txt）：
  llama.cpp 的 llama-perplexity 会说
      "calculating perplexity over N chunks, n_ctx=X"
  当 N × X < 序列长度时，**尾部 token 被直接丢弃**；
  且 `Final estimate: PPL = ...` 取的是**最后一块**，不是全序列平均。

我们此前的对比是「llama.cpp 的最后一块」 vs「PyTorch 的全序列平均」，
两个不同量纲的东西，自然对不上。

本脚本对同一份文本、同一套切块规则，逐块计算 PyTorch loss，
与 llama.cpp 报告的每个块的值一一对照。

用法：python probe_ppl_blocks.py
"""

from __future__ import annotations

import json
import re
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


def get_text(n_tokens: int, tok, skip: int = 8000) -> str:
    parts, n = [], 0
    with (DATA / "pretrain_zh.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            t = json.loads(line)["text"][:600]
            parts.append(t)
            n += len(tok.encode(t).ids)
            if n >= n_tokens:
                break
    return "\n".join(parts)


@torch.no_grad()
def per_chunk_loss(model, ids, chunk, device):
    """按 llama.cpp 的切块规则逐块算 loss。

    llama.cpp 的分词结果首元素是 BOS，然后按 n_ctx 切块；
    第 k 块 = ids[k*chunk : (k+1)*chunk]，块内从第 2 个 token 起算 loss。
    """
    out = []
    for k in range(0, len(ids), chunk):
        piece = ids[k:k + chunk]
        if len(piece) < 2:
            break
        x = torch.tensor([piece], device=device)
        logits = model(x)["logits"].float()
        tgt = x[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), tgt.reshape(-1))
        out.append((k // chunk, len(piece), float(loss)))
    return out


def main() -> None:
    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    CHUNK = 100
    TOKENS = 397          # 与上次受控实验完全相同的文本规模

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    bos = meta["bos"]

    text = get_text(TOKENS, tok)
    ids = [bos] + tok.encode(text).ids
    log(f"序列 {len(ids)} token（含 BOS），切块 {CHUNK} -> "
        f"{int(np.ceil(len(ids)/CHUNK))} 块（末尾 {len(ids) - int(np.ceil(len(ids)/CHUNK))*CHUNK} token 会被 llama.cpp 丢弃）")

    # llama.cpp 的实际输出（上次实验已生成，复用保证文本完全一致）
    raw = (LOGS / "proto_raw.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[1\]([\d.]+),\[2\]([\d.]+),\[3\]([\d.]+)", raw)
    llama_ppls = [float(x) for x in m.groups()] if m else []
    n_chunks_llama = len(llama_ppls)

    # ---- PyTorch ----
    ck = torch.load(str(ROOT / "runs/pretrain/snap500.pt"),
                    map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiMoLite(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    for l in model.layers:
        l.attn.is_global = True

    blocks = per_chunk_loss(model, ids, CHUNK, device)

    print()
    print("=" * 78)
    print(f"{'块号':<8}{'token数':<10}{'PyTorch loss':>16}{'PyTorch ppl':>14}"
          f"{'llama.cpp ppl':>16}{'llama loss':>14}{'差异':>10}")
    print("-" * 78)
    for (idx, n, pl), lp in zip(blocks, llama_ppls):
        ll = float(np.log(lp))
        print(f"{idx+1:<8}{n:<10}{pl:>16.4f}{np.exp(pl):>14.2f}"
              f"{lp:>16.2f}{ll:>14.4f}{ll-pl:>+10.4f}")
    print("=" * 78)

    if blocks and llama_ppls:
        diffs = [float(np.log(lp)) - pl for (_, _, pl), lp in zip(blocks, llama_ppls)]
        print()
        print(f"逐块差异：{[round(d, 4) for d in diffs]}")
        print(f"平均绝对差异：{np.mean(np.abs(diffs)):.4f}")
        print()
        if np.mean(np.abs(diffs)) < 0.05:
            print("✅ 结论：逐块对齐后差异很小（<0.05 nats）。")
            print("   之前的 0.22 nats **不是模型不一致**，而是我们把")
            print("   「llama.cpp 的最后一块」和「PyTorch 的全序列平均」放在一起比。")
            print("   GGUF 导出链路是忠实的。")
        else:
            print("⚠️ 逐块对齐后仍有稳定差异，说明确有实现不一致，继续排查。")
    else:
        print("未能解析 llama.cpp 的逐块输出，请检查 logs/proto_raw.txt")


if __name__ == "__main__":
    main()