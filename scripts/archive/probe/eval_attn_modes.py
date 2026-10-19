"""
SWA vs 全注意力：推理期注意力模式的损失对比
============================================

为什么需要这个脚本？
  MiMo-Lite 的全局注意力层**交错**在第 5、11 层（非"前 N 层"式连续）。
  而 llama.cpp 对 qwen3 架构的滑动窗口语义是"前 n_swa 层使用窗口"。
  导出时写了 qwen3.swa_nums / ga_nums 记录真实层号，但 llama.cpp 是否识别这些键
  未知。两种可能的推理行为：
    A. 识别 swa_nums -> 与训练一致，损失应等于 SWA
    B. 忽略，只按 sliding_window 处理 -> 行为不定
    C. 完全按全注意力 -> 损失接近 full 模式

  先在 PyTorch 里量出「训练同分布(SWA)」与「全注意力」两条基准线，
  再用 llama.cpp 实测输出落在哪条线上，就能反推 llama.cpp 实际用了哪种。

用法：python eval_attn_modes.py --ckpt runs/pretrain/last.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def eval_loss(model, sampler, n_batches: int, seq: int, bs: int, ratio, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x = sampler.batch(bs, ratio).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[:, :-1], labels=x[:, 1:], use_mtp=False)
        tot += out["loss"].item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/pretrain/last.pt"))
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    from model import MiMoLite, MiMoLiteConfig
    from pretrain import PackedSampler

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MiMoLiteConfig(**ck["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiMoLite(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    bins = {"zh": DATA / "packed_zh.bin", "en": DATA / "packed_en.bin"}
    sampler = PackedSampler(bins, args.seq, seed=args.seed)
    ratio = {"zh": 0.7, "en": 0.3}

    log(f"checkpoint: {args.ckpt}  (step={ck.get('step')})")
    log(f"SWA 层: {cfg.swa_nums}")
    log(f"GA  层: {[i for i in range(cfg.n_layer) if cfg.is_global_layer(i)]}")

    # 固定同一批数据，保证两种模式完全可比
    batches = [sampler.batch(args.bs, ratio) for _ in range(args.batches)]

    @torch.no_grad()
    def run(mode: str) -> float:
        if mode == "swa":
            for layer in model.layers:
                layer.attn.is_global = cfg.is_global_layer(
                    int(layer.attn.layer_idx))
        else:  # 全部改成全局注意力
            for layer in model.layers:
                layer.attn.is_global = True
        tot, n = 0.0, 0
        for x in batches:
            x = x.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x[:, :-1], labels=x[:, 1:], use_mtp=False)
            tot += out["loss"].item()
            n += 1
        return tot / n

    swa_loss = run("swa")
    full_loss = run("full")

    print()
    print("=" * 62)
    print(f"{'推理模式':<26}{'验证 loss':>12}")
    print("-" * 62)
    print(f"{'SWA（与训练一致）':<24}{swa_loss:>12.4f}")
    print(f"{'全注意力':<26}{full_loss:>12.4f}")
    print("-" * 62)
    print(f"{'两者差距':<26}{full_loss - swa_loss:>+12.4f}")
    print("=" * 62)
    print()
    print("解读：")
    print(f"  · 差距 < 0.05  -> 两种模式几乎等价，导出成 full 最省心")
    print(f"  · 差距较大      -> 说明模型确实依赖 SWA 的位置先验，")
    print(f"                   应优先确认 llama.cpp 是否识别 qwen3.swa_nums")
    print()
    print("下一步：用 export_gguf.py 分别导出 --attn-mode swa / full 两个 GGUF，")
    print("用 llama.cpp 跑同一段 prompt，看哪条线的 loss/输出更贴近上面哪一行。")


if __name__ == "__main__":
    main()