"""
导出保真度定量检验：PyTorch 困惑度 vs llama.cpp 困惑度
=======================================================

逐 token 对比生成结果在小模型上容易误判——step 500 的模型 logits 很平坦，
bf16 与 f16 的微小数值差异就可能翻转 argmax。
困惑度是连续量，能真实反映两条实现路径的差距。

流程：
  1. 从语料抽一段固定中文文本，逐 token 用 HF tokenizer 编码
  2. PyTorch 侧算 loss
  3. 写纯文本文件，llama-perplexity.exe 算同样的 loss
  4. 对比

用法： python check_export_fidelity.py --gguf out/xxx.gguf --force-full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sample_text(n_chars: int = 1500) -> str:
    """从中文语料里取一段连续文本，保证有代表性。"""
    with (DATA / "pretrain_zh.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 5000:
                continue
            t = json.loads(line)["text"]
            if len(t) > n_chars:
                return t[:n_chars]
    raise RuntimeError("语料不足")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/pretrain/snap500.pt"))
    ap.add_argument("--gguf", type=str, default=str(ROOT / "out/mimolite-500-full.gguf"))
    ap.add_argument("--force-full", action="store_true")
    ap.add_argument("--chars", type=int, default=1500)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    text = sample_text(args.chars)
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    ids = [meta["bos"]] + tok.encode(text).ids
    log(f"测试文本 {len(text)} 字 -> {len(ids)} token")

    # ---- PyTorch 侧 ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = MiMoLite(cfg).to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    if args.force_full:
        for layer in m.layers:
            layer.attn.is_global = True

    seq = torch.tensor([ids], device=device)
    tot_loss, n_tok = 0.0, 0
    # 必须与 llama-perplexity 的 -c 保持一致，否则上下文长度不同会直接影响 loss
    with torch.no_grad():
        for s in range(0, len(ids) - 1, 512):
            chunk = seq[:, s:s + 513]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                out = m(chunk[:, :-1], labels=chunk[:, 1:], use_mtp=False)
            k = int((chunk[:, 1:] != -100).sum())
            tot_loss += out["loss"].item() * k
            n_tok += k
    pt_loss = tot_loss / n_tok
    log(f"PyTorch  loss = {pt_loss:.4f}   ppl = {np.exp(pt_loss):.3f}")

    # ---- llama.cpp 侧 ----
    # 注意：这套 llama.cpp 是窄字符版，对含中文的绝对路径打不开文件
    # （llama-bench / llama-perplexity 都因此失败）。
    # 解决办法：把工作目录设到目标文件夹，只传 ASCII 的相对文件名。
    workdir = ROOT / "logs"
    txt = workdir / "_fidelity.txt"
    txt.write_text(text, encoding="utf-8")
    out = workdir / "_fidelity_llama.txt"
    # 模型路径给绝对路径（llama.cpp 能用宽字符 API 打开模型），
    # 文本文件用 cwd + 相对 ASCII 名（窄字符 API 只能这样绕开中文路径）。
    gguf_abs = str(Path(args.gguf).resolve())
    cmd = [r"F:\AI\tools\llama.cpp\llama-perplexity.exe",
           "-m", gguf_abs, "-f", "_fidelity.txt", "-c", "512", "-b", "512"]
    with out.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, cwd=str(workdir), timeout=900)
    raw = out.read_text(encoding="utf-8", errors="replace")
    (ROOT / "logs" / "fidelity_raw.txt").write_text(raw, encoding="utf-8")

    lp = None
    for line in raw.splitlines():
        low = line.lower()
        # llama.cpp 输出形如：
        #   [1]2225.4726,[2]1881.0280,
        #   Final estimate: PPL = 1881.0280 +/- 164.98617
        if "final estimate" in low or "ppl" in low.replace(" ", ""):
            for tok_s in line.replace("=", " ").replace(",", " ").split():
                try:
                    lp = float(tok_s)
                    break
                except ValueError:
                    pass
            if lp:
                break

    print()
    print("=" * 66)
    print(f"{'':22}{'loss':>10}{'ppl':>12}")
    print("-" * 66)
    print(f"{'PyTorch (bf16 autocast)':<22}{pt_loss:>10.4f}{np.exp(pt_loss):>12.3f}")
    if lp:
        ll = float(np.log(lp))
        print(f"{'llama.cpp (f16)':<22}{ll:>10.4f}{lp:>12.3f}")
        print("-" * 66)
        d = ll - pt_loss
        print(f"{'差距':<22}{d:>+10.4f}")
        if abs(d) < 0.02:
            print("\n结论：两条路径基本等价，GGUF 导出忠实。")
            print("      生成结果的差异主要来自 bf16/f16 数值精度，")
            print("      在 logits 很平坦的小模型上会被 argmax 放大。")
        else:
            print("\n结论：差距过大，导出链路存在问题，需要继续排查。")
    else:
        print(f"{'llama.cpp':<22}{'(解析失败)':>10}")
        print("\n原始输出见 logs/fidelity_raw.txt")
    print("=" * 66)


if __name__ == "__main__":
    main()