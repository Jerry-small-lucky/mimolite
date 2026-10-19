"""
确定性对比：PyTorch 推理 vs llama.cpp 推理
=========================================

这是整条链路是否"忠实"的决定性检验。

导出 GGUF 有太多可能出错的环节：RoPE 风格、张量名映射、tokenizer 序列化、
注意力掩码语义……任何一个错了，llama.cpp 都会静默输出乱码而不报错。

做法：同一份权重、同一个 prompt、两边都用**贪心解码**（temperature=0），
输出必须逐 token 一致。

用法：
  python verify_parity.py --ckpt runs/pretrain/snap500.pt --prompt "深度学习是" --n 24
  python verify_parity.py --ckpt ... --gguf out/xxx.gguf --skip-torch   # 只跑 llama.cpp
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
LLAMA = r"F:\AI\tools\llama.cpp\llama-cli.exe"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def torch_greedy(ckpt: str, prompt: str, n: int, force_full: bool = False) -> str:
    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = MiMoLite(cfg).to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    if force_full:
        # 全部改成全局注意力，对应 GGUF 的 --attn-mode full
        for layer in m.layers:
            layer.attn.is_global = True
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = ck["tokenizer_meta"]

    ids = [meta["bos"]] + tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = m.generate(x, max_new_tokens=n, temperature=0.0,
                         eos_id=meta["eos"], repetition_penalty=1.0,
                         max_context=1024)
    gen = out[0][len(ids):].tolist()
    if meta["eos"] in gen:
        gen = gen[:gen.index(meta["eos"])]
    return tok.decode(gen), gen


def llama_greedy(gguf: str, prompt: str, n: int) -> str:
    tmp = ROOT / "logs" / "_parity_llama.txt"
    cmd = [LLAMA, "-m", str(gguf), "-p", prompt, "-n", str(n),
           "--temp", "0", "--seed", "1", "-ngl", "99", "--no-display-prompt",
           "--log-disable"]
    # llama-cli 生成完会回到交互提示符等待输入，必须给 stdin 接 DEVNULL 让它读到 EOF，
    # 否则进程会一直挂着。
    with open(tmp, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, timeout=600)
    text = tmp.read_text(encoding="utf-8", errors="replace")
    # 取第一段非空行作为输出
    for line in text.splitlines():
        if line.startswith("> "):
            line = line[2:]
        if line.strip():
            return line
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/pretrain/snap500.pt"))
    ap.add_argument("--gguf", type=str, default=str(ROOT / "out/mimolite-500-swa.gguf"))
    ap.add_argument("--prompt", type=str, default="深度学习是")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--force-full", action="store_true",
                    help="PyTorch 侧也强制全注意力（用于与 --attn-mode full 的 GGUF 对比）")
    args = ap.parse_args()

    # llama-cli 会产生大量控制台输出，直接吞掉本进程 stdout 会导致输出丢失，
    # 因此把结论写入报告文件。
    report = ROOT / "logs" / "parity_report.txt"
    lines: list[str] = []

    def emit(m: str) -> None:
        print(m, flush=True)
        lines.append(m)

    emit(f"prompt = {args.prompt!r}   n = {args.n}   贪心解码   force_full={args.force_full}")

    t_out = None
    if not args.skip_torch:
        emit("=== PyTorch 侧 ===")
        t_out, t_ids = torch_greedy(args.ckpt, args.prompt, args.n,
                                    force_full=args.force_full)
        emit(f"输出: {t_out!r}")
        emit(f"token ids: {t_ids}")

    emit("=== llama.cpp 侧 ===")
    l_out = llama_greedy(args.gguf, args.prompt, args.n)
    emit(f"输出: {l_out!r}")

    emit("")
    emit("=" * 66)
    if t_out is None:
        emit("（跳过了 PyTorch 侧，仅记录 llama.cpp 输出）")
    else:
        same = t_out.strip() == l_out.strip()
        emit(f"PyTorch  : {t_out!r}")
        emit(f"llama.cpp: {l_out!r}")
        emit("-" * 66)
        if same:
            emit("一致 —— GGUF 导出链路忠实（RoPE/张量/tokenizer 全部对齐）")
        else:
            emit("输出不一致，需要排查：")
            emit("   1. RoPE 风格（必须 interleaved/NORM）")
            emit("   2. 注意力掩码语义（causal / sliding window）")
            emit("   3. tokenizer 序列化（merges / token scores）")
            emit("   4. 权重 tying（token_embd 是否被当作输出层）")
    emit("=" * 66)

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[报告已写入] {report}")


if __name__ == "__main__":
    main()