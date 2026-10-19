"""
阶段一：从零预训练 MiMo-Lite Base 模型
======================================

对照小米 MiMo 的预训练做法（MiMo-7B / MiMo-V2-Flash 技术报告）：
  - 三阶段数据混合策略 → 本实现按 batch 动态混合中/英文（默认 75:25）
  - Multi-Token Prediction 辅助目标 → MTP loss，权重可调
  - 滑动窗口 + 全局注意力混合 → 直接建在架构里
  - 原生长上下文训练 → 本机 seq_len=512 起步

数据管线：
  jsonl 语料 -> tokenize -> uint16 memmap (.bin) -> 训练时按比例采样打包块

用法：
  python pretrain.py --prepare                 # 仅构建 token 缓存
  python pretrain.py --steps 20000              # 完整训练
  python pretrain.py --steps 500 --smoke       # 冒烟测试
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Windows 下 stdout/stderr 默认走控制台代码页（本机 GBK），一旦输出含中文或
# ✓/─ 等符号就会 UnicodeEncodeError 打掉整个训练进程。强制 UTF-8，
# 不依赖调用方有没有设 PYTHONIOENCODING。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
RUNS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

_LOG_FH = None


def log(m: str) -> None:
    """同时写到 stdout 和日志文件。

    不依赖 shell 管道重定向（PowerShell 的 Out-File 会缓冲很久，
    导致长任务看不到实时进度）。
    """
    global _LOG_FH
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        if _LOG_FH is None:
            _LOG_FH = (LOGS / "pretrain.log").open("a", encoding="utf-8")
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 数据：tokenize -> memmap
# ---------------------------------------------------------------------------
def tokenize_to_bin(jsonl_path: Path, bin_path: Path, tok, batch_lines: int = 4000) -> int:
    """把 jsonl 语料 token 化成 uint16 token 流，**流式追加写盘**。

    流式写盘而非全量累积到内存：中文语料 1.3GB / 约 3 亿 token，
    全量驻留内存既慢又有 OOM 风险。
    """
    if bin_path.exists() and bin_path.stat().st_size > 0:
        n = bin_path.stat().st_size // 2
        log(f"{bin_path.name} 已存在：{n:,} tokens，跳过构建")
        return n

    eos = tok.token_to_id("<|eos|>")
    log(f"tokenize {jsonl_path.name} -> {bin_path.name}")
    total = 0
    t0 = time.time()

    with bin_path.open("wb") as fout, jsonl_path.open(encoding="utf-8") as f:
        buf: list[str] = []
        for line in f:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            if text:
                buf.append(text)
            if len(buf) >= batch_lines:
                encs = tok.encode_batch(buf)
                ids: list[int] = []
                for enc in encs:
                    ids.extend(enc.ids)
                    ids.append(eos)
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(fout)
                total += len(arr)
                buf = []
                if total % 5_000_000 < 100_000:
                    el = time.time() - t0
                    log(f"  {bin_path.name}: {total:,} tokens "
                        f"({total/max(el,1):,.0f} tok/s, {el/60:.1f} 分钟)")
        if buf:
            encs = tok.encode_batch(buf)
            ids = []
            for enc in encs:
                ids.extend(enc.ids)
                ids.append(eos)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(fout)
            total += len(arr)

    log(f"{bin_path.name} 写入完成：{total:,} tokens，耗时 {(time.time()-t0)/60:.1f} 分钟")
    return total


class PackedSampler:
    """
    按比例从多个 token 流中随机取 seq_len+1 的连续块。

    **必须划分 train/val**：消融实验要求验证集与训练集不重叠，
    否则 val loss 退化为训练 loss，任何架构/优化器对比都失去意义。
    切分规则：从每个流的尾部划出 val_ratio 比例作为验证集，
    位置固定、不可变，保证不同 run 之间 val loss 可直接比较。

    验证批次**只生成一次后按 bs 缓存**：如果每次 eval 都从同一个 rng 继续取样，
    rng 会随 eval 次数推进，于是每次验证看到的是不同数据，波动会掩盖真实差异，
    跨 run 也无法比较。
    """

    def __init__(self, bins: dict[str, Path], seq_len: int, seed: int = 0,
                 val_ratio: float = 0.02):
        self.seq_len = seq_len
        self.val_ratio = val_ratio
        self.rng = np.random.default_rng(seed)
        self.val_rng = np.random.default_rng(12345)      # 固定，保证跨 run 可比
        self.arrs: dict[str, np.ndarray] = {}
        self.range: dict[str, tuple[int, int]] = {}      # (train_end, total)
        self.stride: dict[str, int] = {}
        self._val_cache: dict[int, torch.Tensor] = {}    # bs -> 固定验证张量
        for name, path in bins.items():
            if not path.exists():
                continue
            mm = np.memmap(path, dtype=np.uint16, mode="r")
            total = len(mm)
            n_val = max(seq_len * 4, int(total * val_ratio))
            train_end = total - n_val
            self.arrs[name] = mm
            self.range[name] = (train_end, total)
            # 训练：只在前 train_end 里取块（保留 seq_len+1 的余量）
            n_blocks = max(1, (train_end - seq_len - 1) // seq_len)
            self.stride[name] = n_blocks
            log(f"  数据流 {name}: 共 {total:,} tokens | "
                f"train 段 {train_end:,} -> {n_blocks:,} 块 | "
                f"val 段 {n_val:,} tokens（已隔离）")

    def _blocks(self, name: str, arr: np.ndarray, lo: int, hi: int,
                n: int, rng) -> np.ndarray:
        """在 [lo, hi) 区间内取 n 个 seq_len+1 的连续块。"""
        hi = min(hi, len(arr))
        max_start = hi - self.seq_len - 1
        if max_start <= lo:
            return np.empty((0, self.seq_len + 1), dtype=np.int64)
        idx = rng.integers(lo, max_start, size=n)
        return np.stack([arr[int(i):int(i) + self.seq_len + 1] for i in idx])

    def batch(self, bs: int, ratio: dict[str, float], split: str = "train") -> torch.Tensor:
        if split == "val":
            cached = self._val_cache.get(bs)
            if cached is not None:
                return cached

        rng = self.rng if split == "train" else self.val_rng
        names = [n for n in self.arrs if ratio.get(n, 0) > 0]
        # 最大余数法分配：int(bs*ratio) 截断会让实际比例漂移
        # （bs=6, 0.75 → zh 4 / en 1 = 80:20 而非 75:25），
        # 且 ratio*bs < 1 时会给次要语种分配 0 条并直接崩溃。
        counts = _allocate(bs, {n: ratio[n] for n in names})

        xs = []
        for name in names:
            train_end, total = self.range[name]
            if split == "train":
                rows = self._blocks(name, self.arrs[name], 0, train_end,
                                    counts[name], rng)
            else:
                # 验证集从尾部一段里取，区间同样随机但种子固定
                span = max(self.seq_len * 2, int((total - train_end) * 0.8))
                rows = self._blocks(name, self.arrs[name],
                                    total - span, total,
                                    counts[name], rng)
            if len(rows):
                xs.append(rows)
        if not xs:
            raise RuntimeError(f"{split} 集为空，请检查 val_ratio 与数据量")
        x = np.concatenate(xs, axis=0)
        if len(x) < bs:
            pad = x[rng.integers(0, len(x), size=bs - len(x))]
            x = np.concatenate([x, pad], axis=0)
        perm = rng.permutation(len(x))[:bs]
        out = torch.from_numpy(x[perm].astype(np.int64))
        if split == "val":
            self._val_cache[bs] = out
        return out

    # ---- rng 状态：断点续训必须一并保存，否则续训会重放旧数据 ----
    def rng_state(self) -> dict:
        """**只**保存训练流的 rng 状态。

        batch() 每次训练都从 self.rng 抽块，它是一条**有位置**的流：
        抽过 600 步之后再抽，拿到的是新的块。但进程重启后 rng 会回到种子
        初始状态，于是"续跑"的后半程会重新抽到前半程见过的同一批数据。
        19.7M token 的预算里只有约一半是新的 —— 这正是 ablate.py 里
        "等 token 预算"纪律要防的事，故训练 rng 必须进 checkpoint。

        验证流**故意不保存**：验证批次只在进程内缓存一次，跨 run 可比靠的
        正是"每个进程都从固定种子重新抽同一批"。若把 val_rng 恢复到已推进
        过的位置，续跑进程会抽到另一批验证数据，它的 val loss 就不再和其他
        配置处于同一个验证集上 —— 实测会让 val 凭空低 0.7，比配置之间的
        差异大一个数量级，足以把整个消融结论带偏。
        """
        return {"train": self.rng.bit_generator.state}

    def load_rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state["train"]
        # self.val_rng 保持 __init__ 的种子状态：验证集必须每个进程一致。


def _allocate(bs: int, weights: dict[str, float]) -> dict[str, int]:
    """把 bs 行按 weights 精确分配到各流（最大余数法，总和恰为 bs，每项 >=1）。"""
    total_w = sum(weights.values()) or 1.0
    names = list(weights)
    raw = {n: bs * weights[n] / total_w for n in names}
    base = {n: max(1, int(raw[n])) for n in names}
    # 因为 max(1, ...) 可能让总和超过 bs，从大到小回收
    while sum(base.values()) > bs:
        n = max(names, key=lambda x: base[x])
        if base[n] <= 1:
            break
        base[n] -= 1
    # 不足则按小数部分从大到小补齐
    while sum(base.values()) < bs:
        n = max(names, key=lambda x: raw[x] - int(raw[x]))
        base[n] += 1
    return base


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, prog)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def lr_wsd(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.0,
           decay_ratio: float = 0.1) -> float:
    """
    Warmup-Stable-Decay 调度（出处：MiniCPM, arXiv:2404.06395）。

    三段：线性 warmup -> 在峰值保持 -> 最后 decay_ratio 段线性衰减到 min_ratio*peak。
    相比 cosine 的两点优势（本项目直接相关）：
      1. 「稳定段」意味着大部分训练时间都在峰值 LR 上，短预算下比 cosine 走得更远；
      2. 衰减段很短，短预算下衰减不够会吃掉大部分收益 —— 所以 decay_ratio 必须配着短训练步数调。
    GradPower (arXiv:2505.24275) 的最大增益恰好出现在 WSD 调度下。
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    n_decay = max(1, int(total * decay_ratio))
    decay_start = max(warmup, total - n_decay)
    if step < decay_start:
        return peak
    prog = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (min_ratio + (1 - min_ratio) * (1.0 - prog))


def make_lr_fn(kind: str, total: int, peak: float, warmup: int, min_ratio: float,
               decay_ratio: float):
    if kind == "wsd":
        return lambda s: lr_wsd(s, total, peak, warmup, min_ratio, decay_ratio)
    return lambda s: lr_at(s, total, peak, warmup, min_ratio)


def _auto_kv_head(n_head: int, target_group: int = 3) -> int:
    """n_kv_head 取 n_head 的因子中最接近 n_head/target_group 的那个。

    GQA 组大小变了就等于同时改了 KV Cache 压缩率（第二个自变量），
    所以单独跑 pretrain.py 时也沿用与消融框架相同的规则，
    保证两条路径产出的模型可以直接对比。
    """
    divs = [k for k in range(1, n_head + 1) if n_head % k == 0]
    return min(divs, key=lambda v: (abs(n_head // v - target_group), v))


@torch.no_grad()
def gradpower_(params, p: float) -> None:
    """
    GradPower (Wang et al., ICML 2026, arXiv:2505.24275)：
    对梯度做逐元素 sign-power 变换 φ_p(g) = sign(g)·|g|^p，再交给原优化器。

    * 一行改动，不动优化器内部逻辑（动量、二阶矩、超参、weight decay 全部保留）
    * p = 1.2 是论文在 LLaMA-0.2B/C4 上一次定下、随后 66M→2B、dense→MoE、
      C4/OpenWebText、cosine/WSD 全程沿用的默认值
    * 机制：放大"小但持续"的平坦方向、抑制离群 batch 造成的大梯度尖峰
    * 注意 AdamW 对**全局**常数缩放不敏感，真正起作用的是逐元素重新分配相对幅度
    """
    if p == 1.0:
        return
    for prm in params:
        if prm.grad is not None:
            g = prm.grad.data
            # |g|^p 再按 g 的符号恢复方向；p=1 时恒等，故上面直接短路
            prm.grad.data = g.abs().pow_(p).copysign_(g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mtp-weight", type=float, default=0.3)
    ap.add_argument("--zh-ratio", type=float, default=0.75)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--run-name", type=str, default="pretrain")
    ap.add_argument("--zh-data", type=str, default=None,
                    help="中文预训练数据 jsonl（默认 data/pretrain_zh.jsonl；可指定高质量子池）")
    ap.add_argument("--en-data", type=str, default=None,
                    help="英文预训练数据 jsonl（默认 data/pretrain_en.jsonl）")
    ap.add_argument("--prepare", action="store_true", help="只构建 token 缓存")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试配置")
    ap.add_argument("--resume", action="store_true")

    grp = ap.add_argument_group("调度 / 优化器")
    grp.add_argument("--sched", choices=["cosine", "wsd"], default="cosine",
                   help="cosine 或 WSD（MiniCPM arXiv:2404.06395）")
    grp.add_argument("--wsd-decay-ratio", type=float, default=0.1,
                   help="WSD 末段衰减占总步数比例；短训练预算下应调大到 0.2~0.3")
    grp.add_argument("--gradpower-p", type=float, default=1.0,
                   help="GradPower 幂次（arXiv:2505.24275）。1.0=关闭，论文默认 1.2")
    grp.add_argument("--beta2", type=float, default=0.95,
                   help="AdamW β₂（二阶矩衰减）；A5-b 冷段斜坡的起点")
    grp.add_argument("--beta2-cooldown", type=float, default=None,
                   help="若设置，则在训练冷段（最后 wsd_decay_ratio 比例）把 β₂ 从该值斜坡到此处（arXiv:2508.01483）")
    grp.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   help="关闭梯度检查点：省约 25%% 重算，但显存翻倍")
    grp.set_defaults(grad_ckpt=True)

    grp = ap.add_argument_group("架构（消融用，默认即当前基线）")
    grp.add_argument("--n-layer", type=int, default=14)
    grp.add_argument("--d-model", type=int, default=768)
    grp.add_argument("--n-head", type=int, default=0,
                     help="0 = 自动取 d_model/64（与消融框架一致，保证 d_head 恒为 64）")
    grp.add_argument("--n-kv-head", type=int, default=0,
                     help="0 = 自动选 n_head 的因子中最接近 n_head/3 的那个（GQA 组≈3）")
    grp.add_argument("--d-ff", type=int, default=2048)
    grp.add_argument("--window", type=int, default=128)
    grp.add_argument("--ga-interval", type=int, default=6)
    grp.add_argument("--ga-layers", type=str, default="",
                   help="显式 GA 层号，逗号分隔；留空则用 --ga-interval 均匀交错")
    grp.add_argument("--attn-impl", choices=["dense", "flex"], default="dense",
                   help="SWA 掩码实现；flex=FlexAttention block-sparse")
    grp.add_argument("--mtp-depth", type=int, default=1)

    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from tokenizers import Tokenizer
    from model import MiMoLiteConfig, MiMoLite

    if args.smoke:
        args.steps = min(args.steps, 30)
        args.eval_every = 15
        args.save_every = 30
        args.run_name += "_smoke"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    # ---- tokenizer ----
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        log("未找到 tokenizer，请先运行 train_tokenizer.py")
        return
    tok = Tokenizer.from_file(str(tok_path))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    eos_id, pad_id, bos_id = meta["eos"], meta["pad"], meta["bos"]

    # ---- 构建 token 缓存 ----
    # 数据源：默认 data/pretrain_{lang}.jsonl；可经 --zh-data / --en-data 指定
    # 高质量子池（如 data/a001_zh.jsonl），bin 跟随数据文件命名，互不覆盖。
    bins = {"zh": DATA / "packed_zh.bin", "en": DATA / "packed_en.bin"}
    for lang in ("zh", "en"):
        override = getattr(args, f"{lang}_data", None)
        src = Path(override) if override else DATA / f"pretrain_{lang}.jsonl"
        if src.exists():
            bin_path = (
                DATA / f"packed_{lang}.bin" if not override
                else DATA / f"{src.stem}.bin"
            )
            tokenize_to_bin(src, bin_path, tok)
            bins[lang] = bin_path

    if args.prepare:
        log("token 缓存构建完成")
        return

    # ---- 配置 ----
    # d_head 固定 64（RoPE 维度已在 llama.cpp 保真度实验中验证过），
    # n_head 默认由 d_model 推导；显式给 --n-head 时必须整除 d_model。
    if args.n_head and args.d_model % args.n_head:
        raise SystemExit(
            f"--n-head {args.n_head} 不能整除 --d-model {args.d_model}。"
            f"d_head 必须为整数；建议省略 --n-head 让它自动取 d_model/64。")
    n_head = args.n_head or max(1, args.d_model // 64)
    n_kv_head = args.n_kv_head or _auto_kv_head(n_head)
    if n_head % n_kv_head:
        raise SystemExit(
            f"--n-kv-head {n_kv_head} 不能整除 --n-head {n_head}。"
            f"可用值：{[k for k in range(1, n_head+1) if n_head % k == 0]}")
    ga_layers = tuple(int(i) for i in args.ga_layers.split(",") if i.strip() != "") or None
    cfg = MiMoLiteConfig(
        vocab_size=meta["vocab_size"], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        n_layer=args.n_layer, n_head=n_head, n_kv_head=n_kv_head,
        d_model=args.d_model, d_head=args.d_model // n_head, d_ff=args.d_ff,
        window=args.window, ga_interval=args.ga_interval, ga_layers=ga_layers,
        attn_impl=args.attn_impl, mtp_depth=args.mtp_depth,
    )
    model = MiMoLite(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量：{n_params/1e6:.2f}M")
    log(f"GA 层 {cfg.ga_nums} | SWA 层 {cfg.swa_nums} | SWA:GA = {cfg.swa_ga_ratio} "
        f"| window={cfg.window} | attn_impl={cfg.attn_impl} | mtp_depth={cfg.mtp_depth}")

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, fused=True,
    )

    run_dir = RUNS / args.run_name
    run_dir.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"输出目录：{run_dir}")

    sampler = PackedSampler(bins, args.seq_len, seed=1234)
    ratio = {"zh": args.zh_ratio, "en": round(1 - args.zh_ratio, 4)}

    start_step = 0
    restored_hist: list[dict] = []
    restored_val: list[dict] = []
    ckpt_path = run_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "sampler_rng" not in ck:
            raise SystemExit(
                "RESUME_REFUSED: "
                f"{ckpt_path} 里没有数据流 rng 状态，是修复前写下的旧 checkpoint。"
                "用它续跑会让后半程重放前半程见过的数据，等 token 预算直接失效，"
                "因此拒绝续跑。请去掉 --resume 从头跑。")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        log(f"断点续训：step={start_step}")
        # 数据流位置 + 历史曲线一并恢复，否则半截的 run 会丢掉前半程记录，
        # read_result() 会把 loss_curve 的末条 val 当成"这格只跑了这些步"。
        sampler.load_rng_state(ck["sampler_rng"])
        # .cpu() 不能省：ckpt 是用 map_location=device 载入的，torch_rng
        # 会被搬到 GPU 上，而 set_rng_state 只接受 CPU ByteTensor。
        torch.set_rng_state(ck["torch_rng"].cpu())
        np.random.set_state(ck["numpy_rng"])
        restored_hist = list(ck.get("hist", []))
        restored_val = list(ck.get("val_hist", []))
        log(f"恢复数据流 rng 位置 + {len(restored_hist)} 条训练记录"
            f" / {len(restored_val)} 条验证记录")

    tokens_per_step = args.micro_bs * args.accum * (args.seq_len + 1)
    total_tokens = args.steps * tokens_per_step
    log(f"计划：{args.steps:,} steps × {tokens_per_step:,} tok/step = {total_tokens/1e6:.1f}M tokens")
    log(f"micro_bs={args.micro_bs} accum={args.accum} seq={args.seq_len}")

    val_fp: dict = {"id": None}

    @torch.no_grad()
    def evaluate() -> float:
        """固定验证集上的 loss（与训练集严格隔离，且各 run 可直接比较）。"""
        model.eval()
        tot, n = 0.0, 0
        for _ in range(args.eval_batches):
            xb = sampler.batch(args.micro_bs, ratio, split="val")
            # 验证集指纹：只要它变了，这个 run 的 val loss 就不能和别的 run 相减。
            # 曾经踩过 —— 续跑时若把 val_rng 恢复到已推进的位置，进程会抽到
            # 另一批验证数据，val 凭空低 0.7，看起来像"这一格架构更好"。
            if val_fp["id"] is None:
                val_fp["id"] = hashlib.md5(
                    xb.numpy().tobytes()).hexdigest()[:12]
                log(f"验证集指纹 = {val_fp['id']}（跨 run 必须一致才可比）")
            x = xb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(x[:, :-1], labels=x[:, 1:], use_mtp=False)
            tot += out["loss"].item()
            n += 1
        model.train()
        return tot / max(n, 1)

    log(f"初始评估 loss = {evaluate():.4f}")

    model.train()
    t0 = time.time()
    ema = None
    hist = list(restored_hist)
    lr_fn = make_lr_fn(args.sched, args.steps, args.lr, args.warmup,
                       args.min_lr_ratio, args.wsd_decay_ratio)
    train_params = [p for p in model.parameters() if p.requires_grad]
    if args.gradpower_p != 1.0:
        log(f"GradPower 启用：p={args.gradpower_p}（ICML 2026 arXiv:2505.24275 默认 1.2）")
    if args.sched == "wsd":
        log(f"WSD 调度：peak={args.lr:.2e} warmup={args.warmup} "
            f"decay 段={int(args.steps*args.wsd_decay_ratio)} 步 → min={args.min_lr_ratio*args.lr:.2e}")

    val_hist: list[dict] = list(restored_val)
    _curve_path = run_dir / "loss_curve.json"

    def dump_curve() -> None:
        """每次 eval / save 都落盘：被 Ctrl-C / OOM 打断时也不丢实验数据。"""
        try:
            _curve_path.write_text(json.dumps(
                {"args": vars(args), "n_params": n_params,
                 "ga_layers": cfg.ga_nums, "swa_ga": cfg.swa_ga_ratio,
                 "train": hist, "val": val_hist}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # 冷段 β₂ 斜坡（A5-b：WSD cooldown 期提高 β₂，arXiv:2508.01483）
    cooldown_ratio = args.wsd_decay_ratio
    cooldown_start = max(args.warmup, args.steps - max(1, int(args.steps * cooldown_ratio)))
    b2_base = args.beta2
    b2_target = args.beta2_cooldown
    effective = cooldown_start < args.steps
    ramp_b2 = b2_target is not None and b2_target != b2_base and effective
    if ramp_b2:
        log(f"β₂ 斜坡：{b2_base} → {b2_target}（步 {cooldown_start}–{args.steps}，"
            f"冷段比例 {cooldown_ratio:.0%}）")
    elif b2_target is not None and b2_target != b2_base:
        log(f"β₂ 冷段斜坡未启用：steps({args.steps}) ≤ warmup({args.warmup})，无可用冷段")

    for step in range(start_step, args.steps):
        cur_lr = lr_fn(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        if ramp_b2 and step >= cooldown_start:
            prog = min(1.0, (step - cooldown_start) / max(1, args.steps - cooldown_start))
            cur_b2 = b2_base + (b2_target - b2_base) * prog
            for g in opt.param_groups:
                g["betas"] = (0.9, cur_b2)

        opt.zero_grad(set_to_none=True)
        loss_acc = mtp_acc = 0.0
        for micro in range(args.accum):
            x = sampler.batch(args.micro_bs, ratio).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(x[:, :-1], labels=x[:, 1:], use_mtp=True)
            loss = out["loss"] + args.mtp_weight * out.get("mtp_loss", 0.0)
            (loss / args.accum).backward()
            loss_acc += out["loss"].item() / args.accum
            if "mtp_loss" in out:
                mtp_acc += out["mtp_loss"].detach().item() / args.accum

        # GradPower 作用在累积后的梯度上，再进 clip + step
        # （论文的表述是 "feeds the transformed gradient into a base optimizer"，
        #   梯度裁剪属于 base optimizer 管线的一部分）
        gradpower_(train_params, args.gradpower_p)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        ema = loss_acc if ema is None else 0.9 * ema + 0.1 * loss_acc
        hist.append({"step": step, "loss": loss_acc, "mtp_loss": mtp_acc, "lr": cur_lr})

        if (step + 1) % args.log_every == 0:
            el = time.time() - t0
            done = step - start_step + 1
            tps = done * tokens_per_step / el
            eta = (args.steps - step - 1) * tokens_per_step / max(tps, 1)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            log(f"step {step+1:6d}/{args.steps} | loss {loss_acc:.4f} (ema {ema:.4f}) | "
                f"mtp {mtp_acc:.4f} | lr {cur_lr:.2e} | {tps/1000:.1f}k tok/s | "
                f"eta {eta/60:.0f}m | mem {mem:.1f}G | gnorm {float(gn):.2f}")

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vl = evaluate()
            log(f"  >>> step {step+1} 验证 loss = {vl:.4f}")
            hist.append({"step": step, "val_loss": vl})
            val_hist.append({"step": step + 1, "val_loss": vl})
            dump_curve()

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step + 1,
                "config": cfg.to_dict(),
                "tokenizer_meta": meta,
                "hist": hist[-500:],
                "val_hist": val_hist,
                # 数据流位置 + 全部随机源：缺了任何一个，续跑都会悄悄改变
                # 训练条件（rng 缺失=重放数据，torch_rng 缺失=dropout 序列变化）。
                "sampler_rng": sampler.rng_state(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            }, ckpt_path)

    torch.save({
        "model": model.state_dict(), "step": args.steps,
        "config": cfg.to_dict(), "tokenizer_meta": meta, "hist": hist[-2000:],
    }, run_dir / "final.pt")
    dump_curve()
    log(f"训练完成，总耗时 {(time.time()-t0)/60:.1f} 分钟 -> {run_dir/'final.pt'}")
    if val_hist:
        log(f"最终验证 loss = {val_hist[-1]['val_loss']:.4f}（step {val_hist[-1]['step']}）")


if __name__ == "__main__":
    main()
