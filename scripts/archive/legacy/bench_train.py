"""
训练吞吐基准 —— 用合成数据实测本机训练速度，用于确定正式预训练规模
====================================================================

不依赖真实语料（.bin 还没构建完也能跑）。
输出 tokens/s、显存峰值，据此推算：训练 N steps 需要多久。

用法：python bench_train.py
"""

from __future__ import annotations

import time

import torch

from model import MiMoLite, MiMoLiteConfig


def bench(micro_bs: int, accum: int, seq: int, steps: int = 8, ckpt: bool = True):
    cfg = MiMoLiteConfig(vocab_size=16384)
    device = "cuda"
    model = MiMoLite(cfg).to(device)
    if ckpt:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)

    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, cfg.vocab_size, (micro_bs, seq + 1), device=device)

    model.train()
    for i in range(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x[:, :-1], labels=x[:, 1:], use_mtp=True)
                loss = out["loss"] + 0.3 * out["mtp_loss"]
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0

    tokens = micro_bs * accum * (seq + 1) * steps
    tps = tokens / (dt * steps)
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"micro_bs={micro_bs:3d} accum={accum} seq={seq:4d} ckpt={int(ckpt)} | "
          f"{tps/1000:6.1f} k tok/s | 显存峰值 {mem:.2f} GB | step {dt*1000:.0f} ms",
          flush=True)
    del model, opt, x
    torch.cuda.empty_cache()
    return tps, mem


if __name__ == "__main__":
    print("=== RTX 5070 Laptop 8GB 训练吞吐基准 ===")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    results = []
    for mbs, acc, seq, ck in [
        (8, 4, 512, True),
        (8, 4, 512, False),   # 与上一行构成受控对照：唯一变量是梯度检查点
        (16, 4, 512, True),
        (16, 4, 512, False),
        (24, 4, 512, True),
        (32, 2, 512, True),
        (16, 2, 1024, True),
    ]:
        try:
            results.append(((mbs, acc, seq, ck), *bench(mbs, acc, seq, ckpt=ck)))
        except torch.cuda.OutOfMemoryError:
            print(f"micro_bs={mbs} accum={acc} seq={seq} ckpt={int(ck)} -> OOM", flush=True)
            torch.cuda.empty_cache()

    if results:
        best = max(results, key=lambda r: r[1])
        (mbs, acc, seq, ck), tps, mem = best
        print(f"\n最佳配置：micro_bs={mbs} accum={acc} seq={seq} grad_ckpt={ck}")
        print(f"吞吐 {tps/1000:.1f} k tok/s，显存 {mem:.2f} GB")
        print("\n各规模预估时长：")
        for steps, label in [(2000, "冒烟"), (6000, "短训"), (12000, "标准"), (24000, "完整")]:
            toks = steps * mbs * acc * (seq + 1)
            print(f"  {label:4s} {steps:6d} steps = {toks/1e6:7.1f}M tokens "
                  f"-> {toks/tps/3600:5.2f} 小时")
