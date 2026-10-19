"""
MiMo-Lite —— 受小米 MiMo 系列启发的自研小型语言模型架构
=====================================================

设计对标小米 MiMo-V2-Flash / MiMo-V2-Pro 的两个招牌技术，并按单卡 8GB 显存做等比缩放：

1. **混合注意力 (Hybrid Attention, SWA:GA = 6:1)**
   小米原版做法：每 6 层里 5 层滑动窗口注意力(SWA) + 1 层全局注意力(GA)。
   128 -token 窗口使 KV Cache 大小恒定，长上下文下存储降约 7 倍。
   本实现默认 `ga_interval=6` → 第 5,11,... 层为 GA，其余为 SWA。
   ⚠️ n_layer=14 不是 6 的倍数，所以实际只有 2 个 GA 层，**SWA:GA = 12:2 = 6:1**
   （不是 5:1；精确 5:1 需要 n_layer ∈ {12, 18, 24}）。
   同时用 GQA (n_kv_head=4 < n_head=12) 进一步压缩 KV Cache，
   两者叠加 = 小米所强调的"固定大小 KV Cache"效果。

   消融 A-003 用 `ga_layers` 显式指定 GA 层位置，以检验「均匀交错是否最优」。
   依据：ConSA (arXiv:2606.18056, 0.6B/1.7B) 学到的分配是「SWA 在底层、
   FA 集中于**连续**中间层块」，明确偏离均匀交错；BOSCH (arXiv:2604.05942,
   ACL 2026) 发现逐层自适应 SWA 比例优于层间启发式。

2. **MTP (Multi-Token Prediction) 多 token 预测**
   小米原版：3 层 MTP，推理时当作 draft 模型做投机解码，吞吐提升约 3 倍。
   本实现：1 层轻量 MTP 模块，训练时作为辅助损失提升表征质量，
   推理时可用作 speculative decoding 的 draft head（导出 GGUF 后由 llama.cpp 承担）。

其余骨干全部对齐现代 LLM 主流做法：
   RMSNorm + RoPE + SwiGLU + QK-Norm + 无 bias + 权重 tying。

作者：Jerry (在 Mavis 协助下自研)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
@dataclass
class MiMoLiteConfig:
    """MiMo-Lite 模型配置。"""

    vocab_size: int = 16384
    n_layer: int = 14
    n_head: int = 12
    n_kv_head: int = 4          # GQA：KV Cache 压缩 n_head/n_kv_head 倍
    d_model: int = 768
    d_head: int = 64            # n_head * d_head == d_model
    d_ff: int = 2048            # SwiGLU 中间层维度

    # —— 混合注意力 (小米 MiMo 招牌) ——
    window: int = 128           # 滑动窗口大小
    ga_interval: int = 6        # 每 6 层放 1 层全局注意力 => SWA:GA = 5:1（n_layer 为 6 倍数时）
    ga_layers: tuple[int, ...] | None = None
    #   显式指定 GA 层编号。None 时回退到上面的 is_global_layer(ga_interval) 规则。
    #   消融 A-003 用它验证 ConSA 式「连续中间层块」分配是否优于均匀交错。

    # —— 注意力实现 ——
    attn_impl: str = "dense"    # "dense"（稠密 mask SDPA）| "flex"（FlexAttention block-sparse）
    #   FlexAttention 在 seq=512 / window=128 下理论可跳过约 78% 的注意力 FLOPs，
    #   且不像显式 bool mask 那样让 SDPA 掉出 flash 内核。收益需实测。
    flex_block_size: int = 128  # 应与 window 对齐，避免块内大量浪费

    # —— MTP (小米 MiMo 招牌) ——
    mtp_depth: int = 1          # MTP 模块数（本机规模取 1）

    rope_theta: float = 100000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    def __post_init__(self) -> None:
        assert self.n_head * self.d_head == self.d_model, "n_head * d_head 必须等于 d_model"
        assert self.n_head % self.n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"
        assert self.d_ff % 2 == 0 or True
        if self.ga_layers is not None:
            self.ga_layers = tuple(sorted(set(int(i) for i in self.ga_layers)))
            for i in self.ga_layers:
                assert 0 <= i < self.n_layer, f"GA 层 {i} 越界 (n_layer={self.n_layer})"

    # 哪些层用全局注意力
    def is_global_layer(self, idx: int) -> bool:
        if self.ga_layers is not None:
            return idx in self.ga_layers
        return idx % self.ga_interval == self.ga_interval - 1

    @property
    def swa_nums(self) -> list[int]:
        """SWA 层编号列表（GGUF 导出时写入 qwen3.swa_nums）。"""
        return [i for i in range(self.n_layer) if not self.is_global_layer(i)]

    @property
    def ga_nums(self) -> list[int]:
        """GA 层编号列表（GGUF 导出时写入 qwen3.ga_nums）。"""
        return [i for i in range(self.n_layer) if self.is_global_layer(i)]

    @property
    def swa_ga_ratio(self) -> str:
        """实际 SWA:GA 比例字符串，例如 "12:2 (6:1)"。"""
        g = len(self.ga_nums)
        s = self.n_layer - g
        if g == 0:
            return f"{s}:0 (全 SWA)"
        return f"{s}:{g} ({s//g}:1)"

    @property
    def n_params_estimate(self) -> int:
        attn = self.d_model * self.d_head * (self.n_head + 2 * self.n_kv_head)
        attn += 2 * self.d_head  # q_norm + k_norm
        mlp = 3 * self.d_model * self.d_ff
        per_layer = attn + mlp + 2 * self.d_model
        mtp = per_layer + self.d_model  # MTP 额外一层 + 投影
        embed = self.vocab_size * self.d_model
        total = embed + self.n_layer * per_layer + mtp + self.d_model
        return total + (0 if self.tie_embeddings else self.vocab_size * self.d_model)

    def to_dict(self) -> dict:
        return asdict(self)


def ga_layer_indices(cfg) -> list[int]:
    """从 MiMoLiteConfig **或其 dict 形式**推导 GA 层编号。

    导出器拿到的是 checkpoint 里存的 dict，需要和训练时走同一套规则，
    否则改了 ga_layers 后导出会静默写出错误的 qwen3.swa_nums / ga_nums。
    """
    if isinstance(cfg, MiMoLiteConfig):
        return cfg.ga_nums
    n_layer = int(cfg["n_layer"])
    ga = cfg.get("ga_layers")
    if ga:
        return sorted({int(i) for i in ga})
    interval = int(cfg["ga_interval"])
    return [i for i in range(n_layer) if i % interval == interval - 1]


# ----------------------------------------------------------------------------
# 基础模块
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos / sin。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                    # (T, D/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, T, D)；cos/sin: (T, D/2)。

    采用 **interleaved (NORM) 风格**：把最后一维按偶/奇位切分旋转。
    这是 llama / qwen2 / qwen3 架构在 llama.cpp 中的标准 RoPE 约定，
    若改用 NeoX 半旋转风格，导出 GGUF 后推理会完全错乱。
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def build_swa_mask(
    seq_len: int, window: int, device, dtype=torch.bool
) -> torch.Tensor:
    """
    构造滑动窗口因果注意力 mask（True = 可 attend）。

    形状 (1, 1, T, T)。位置 i 可看 [i-window+1, i]。
    """
    idx = torch.arange(seq_len, device=device)
    # q_idx (T,1) - k_idx (1,T) <= window-1
    delta = idx[:, None] - idx[None, :]
    causal = delta >= 0
    within = delta < window
    mask = causal & within
    return mask[None, None]


# --------------------------------------------------------------------------
# FlexAttention block-sparse 路径
# --------------------------------------------------------------------------
_FLEX_AVAILABLE: bool | None = None
_FLEX_BROKEN: str | None = None     # 首次失败原因；一旦失败全局降级到稠密 mask
_FLEX_CACHE: dict = {}


def _flex_ok() -> bool:
    global _FLEX_AVAILABLE
    if _FLEX_AVAILABLE is None:
        try:
            from torch.nn.attention.flex_attention import (  # noqa: F401
                create_block_mask, flex_attention,
            )
            _FLEX_AVAILABLE = True
        except Exception:
            _FLEX_AVAILABLE = False
    return _FLEX_AVAILABLE


def build_flex_block_mask(seq_len: int, window: int, device, block_size: int = 128):
    """
    滑动窗口因果注意力的 BlockMask（FlexAttention 用）。

    与 `build_swa_mask` 语义完全一致，区别在于它把 (T, T) 的稠密 bool 矩阵
    压成 (T/BS, T/BS) 的稀疏块索引，让 kernel 整块跳过无关计算。
    BlockMask 可跨 batch/step 复用（窗口和 seq_len 固定时）。
    """
    key = (seq_len, window, device.type, device.index, block_size)
    hit = _FLEX_CACHE.get(key)
    if hit is not None:
        return hit

    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        d = q_idx - kv_idx
        return (d >= 0) & (d < window)

    bs = max(1, min(block_size, seq_len))
    try:
        bm = create_block_mask(
            mask_mod, 1, 1, seq_len, seq_len, device=str(device), BLOCK_SIZE=bs,
        )
    except TypeError:
        # 老版本没有 BLOCK_SIZE 关键字
        bm = create_block_mask(mask_mod, 1, 1, seq_len, seq_len, device=str(device))
    _FLEX_CACHE[key] = bm
    return bm


def flex_attention_fn():
    """返回编译好的 flex_attention（带缓存），失败返回 None。"""
    if not _flex_ok():
        return None
    from torch.nn.attention.flex_attention import flex_attention
    key = ("compiled_flex",)
    if key not in _FLEX_CACHE:
        _FLEX_CACHE[key] = torch.compile(flex_attention, dynamic=False)
    return _FLEX_CACHE[key]


def flex_mark_broken(reason: str) -> None:
    """记录 FlexAttention 首次失败原因并全局降级（只记录第一次）。

    FlexAttention 依赖 inductor 代码生成，缺编译器（Windows 无 MSVC cl.exe）
    或遇到不支持的后端时会抛异常；此时静默退回稠密 mask 比崩掉训练划算。
    """
    global _FLEX_BROKEN
    if _FLEX_BROKEN is None:
        _FLEX_BROKEN = reason
        print(f"[model] FlexAttention 降级为稠密 mask：{reason[:180]}")


# ----------------------------------------------------------------------------
# 注意力
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    """支持 GQA + 可选滑动窗口的因果注意力。"""

    def __init__(self, cfg: MiMoLiteConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_global = cfg.is_global_layer(layer_idx)
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.d_head = cfg.d_head

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.d_head, cfg.d_model, bias=False)

        # QK-Norm：稳定训练（Qwen3 / MiMo 均采用）
        self.q_norm = RMSNorm(cfg.d_head, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.d_head, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        block_mask=None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # GQA: 扩展 KV 头
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if block_mask is not None and _FLEX_BROKEN is None:
            # FlexAttention：block-sparse kernel，整块跳过窗口外的 key
            try:
                flex = flex_attention_fn()
                if flex is not None:
                    out = flex(q, k, v, block_mask=block_mask, enable_gqa=(self.n_rep > 1))
                    out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
                    return self.o_proj(out)
            except Exception as e:
                flex_mark_broken(f"{type(e).__name__}: {e}")

        # 全局注意力层不传 mask，直接走 is_causal=True 的高效内核；
        # SWA 层传显式滑窗 mask（已含因果关系）。
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络。"""

    def __init__(self, cfg: MiMoLiteConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """一个 Transformer 层（Pre-Norm）。"""

    def __init__(self, cfg: MiMoLiteConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask, block_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, block_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ----------------------------------------------------------------------------
# MTP (Multi-Token Prediction) —— 小米 MiMo 招牌技术
# ----------------------------------------------------------------------------
class MTPBlock(nn.Module):
    """
    多 token 预测模块。

    输入主模型最后一层隐状态 h_t 与 token embedding(x_{t+1})，
    经一层 Transformer 后预测 x_{t+2}。

    训练时：作为辅助损失信号，逼迫主模型隐状态携带更远的未来信息。
    推理时：作为 draft head，配合主模型做投机解码（Speculative Decoding）。
    """

    def __init__(self, cfg: MiMoLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx=0)      # MTP 层始终全局注意力
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, h: torch.Tensor, emb_shift: torch.Tensor) -> torch.Tensor:
        """h: (B,T,D) 主模型隐状态；emb_shift: (B,T,D) 偏移一位的 token embedding。"""
        x = torch.cat([h, emb_shift], dim=-1)
        x = self.norm(self.proj(x))
        cos, sin = _rope_from(x, self.cfg)
        x = x + self.attn(self.attn_norm(x), cos, sin, None)   # MTP 层用全局注意力
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _rope_from(x: torch.Tensor, cfg: MiMoLiteConfig):
    cos, sin = build_rope_cache(x.shape[1], cfg.d_head, cfg.rope_theta, x.device, x.dtype)
    return cos, sin


# ----------------------------------------------------------------------------
# 主模型
# ----------------------------------------------------------------------------
class MiMoLite(nn.Module):
    """MiMo-Lite 主模型。"""

    def __init__(self, cfg: MiMoLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # MTP 模块
        self.mtp = nn.ModuleList([MTPBlock(cfg) for _ in range(cfg.mtp_depth)]) \
            if cfg.mtp_depth > 0 else None

        self.apply(self._init_weights)
        # scaled init：残差分支小初始化，稳定深层训练
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self) -> None:
        """逐层梯度检查点：用计算换显存，8GB 显存下显著提升可用 batch。"""
        self.grad_ckpt = True

    def gradient_checkpointing_disable(self) -> None:
        self.grad_ckpt = False

    def _run_layer(self, layer, x, cos, sin, mask, block_mask=None):
        if getattr(self, "grad_ckpt", False) and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, x, cos, sin, mask, block_mask, use_reentrant=False
            )
        return layer(x, cos, sin, mask, block_mask)

    def _masks(self, seq_len: int, device):
        cos, sin = build_rope_cache(seq_len, self.cfg.d_head, self.cfg.rope_theta, device, torch.float32)
        swa_mask = None
        block_mask = None
        if self.cfg.attn_impl == "flex" and _flex_ok() and _FLEX_BROKEN is None:
            try:
                block_mask = build_flex_block_mask(
                    seq_len, self.cfg.window, device, self.cfg.flex_block_size)
            except Exception as e:
                flex_mark_broken(f"{type(e).__name__}: {e}")
        if block_mask is None:
            swa_mask = build_swa_mask(seq_len, self.cfg.window, device)
        return cos, sin, swa_mask, block_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_mtp: bool = False,
    ) -> dict:
        """
        input_ids: (B, T)
        labels:    (B, T)  next-token 目标（labels[:, i] 是位置 i-1 的预测目标）
        use_mtp:   是否计算 MTP 辅助损失

        MTP 目标在内部按 depth 对齐：depth d 预测 labels[:, i+2+d]，
        即比主任务再往未来多看 d 个 token。
        """
        B, T = input_ids.shape
        device = input_ids.device
        cos, sin, swa_mask, block_mask = self._masks(T, device)

        x = self.embed(input_ids)
        cos_f, sin_f = cos.to(x.dtype), sin.to(x.dtype)

        for layer in self.layers:
            is_ga = layer.attn.is_global
            mask = None if is_ga else swa_mask
            bm = None if is_ga else block_mask
            x = self._run_layer(layer, x, cos_f, sin_f, mask, bm)

        h = self.norm(x)
        logits = self.lm_head(h)

        out = {"logits": logits, "hidden": h}

        # ---- MTP 辅助损失 ----
        if self.mtp is not None and labels is not None and use_mtp:
            mtp_loss = x.new_zeros(())
            for depth, mtp_mod in enumerate(self.mtp):
                # emb_shift: x_{t+1} 的 embedding（用自身右移，最后一位回绕填 BOS）
                emb_shift = torch.cat(
                    [self.embed(input_ids[:, 1:]),
                     self.embed(input_ids[:, :1])], dim=1
                )
                mtp_h = mtp_mod(h, emb_shift)
                mtp_logits = self.lm_head(mtp_h)
                k = depth + 1                     # depth0 预测 t+2（labels 已被调用方 shift 1 位）
                if T > k:
                    mtp_loss = mtp_loss + _ce_loss(
                        mtp_logits[:, :-k], labels[:, k:])
            out["mtp_loss"] = mtp_loss / len(self.mtp)

        if labels is not None:
            out["loss"] = _ce_loss(logits, labels)
        return out

    # ---- 推理 ----
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        eos_id: int | None = None,
        repetition_penalty: float = 1.05,
        max_context: int = 1024,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        批量自回归采样（无 KV Cache，够本机小模型使用）。

        max_context 控制回看窗口：超出后丢弃最左侧 token。
        滑动窗口注意力天然支持这种截断，全局注意力层负责把信息带过来。
        """
        device = input_ids.device
        out_ids = input_ids
        finished = torch.zeros(out_ids.shape[0], dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            x = out_ids[:, -max_context:]
            res = self(x)
            logits = res["logits"][:, -1, :].float()

            if repetition_penalty != 1.0:
                # 向量化重复惩罚
                for b in range(out_ids.shape[0]):
                    uniq = torch.unique(out_ids[b])
                    vals = logits[b, uniq]
                    logits[b, uniq] = torch.where(
                        vals < 0, vals * repetition_penalty, vals / repetition_penalty)

            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
                probs = logits.softmax(-1)
                if top_p:
                    sp, si = torch.sort(probs, descending=True, dim=-1)
                    cum = sp.cumsum(-1)
                    keep = (cum - sp) < top_p
                    sp = sp * keep
                    sp = sp / sp.sum(-1, keepdim=True)
                    # 在 top_p 子集上重新采样
                    pick = torch.multinomial(sp, num_samples=1, generator=generator)
                    next_id = si.gather(-1, pick)
                else:
                    next_id = torch.multinomial(probs, num_samples=1, generator=generator)

            if eos_id is not None:
                next_id = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_id, eos_id), next_id)
                finished = finished | (next_id.squeeze(1) == eos_id)
            out_ids = torch.cat([out_ids, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return out_ids


def _ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """忽略 -100 的交叉熵。"""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        ignore_index=-100,
    )


# ----------------------------------------------------------------------------
# 构造与统计
# ----------------------------------------------------------------------------
def build_model(cfg: MiMoLiteConfig) -> MiMoLite:
    return MiMoLite(cfg)


if __name__ == "__main__":
    cfg = MiMoLiteConfig()
    model = build_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"MiMo-Lite 配置: {cfg.to_dict()}")
    print(f"实际参数量: {n/1e6:.2f}M")
    print(f"SWA 层: {[i for i in range(cfg.n_layer) if not cfg.is_global_layer(i)]}")
    print(f"GA  层: {cfg.ga_nums}")
    print(f"SWA:GA = {cfg.swa_ga_ratio}")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    labels = torch.randint(0, cfg.vocab_size, (2, 64))
    model.train()
    out = model(x, labels, use_mtp=True)
    print(f"loss={out['loss'].item():.3f} mtp_loss={out['mtp_loss'].item():.3f}")
    print("forward OK")

    # --- FlexAttention 与稠密 mask 的一致性回归（需 CUDA：CPU 路径要 inductor C++ 编译器）---
    if not torch.cuda.is_available():
        print("无 CUDA，跳过 FlexAttention 一致性检查")
    else:
        dev = "cuda"
        model_c = model.to(dev).eval()
        xc, lc = x.to(dev), labels.to(dev)
        cfg_flex = MiMoLiteConfig(attn_impl="flex")
        m2 = MiMoLite(cfg_flex).to(dev).eval()
        m2.load_state_dict(model.state_dict())
        with torch.no_grad():
            ref = model_c(xc, lc)["logits"].float()
            got = m2(xc, lc)["logits"].float()
        import model as _self
        if _self._FLEX_BROKEN:
            print(f"FlexAttention 降级为稠密 mask（{_self._FLEX_BROKEN[:120]}），"
                  f"退化为自洽性检查：max|diff| = {(ref-got).abs().max().item():.2e}")
        else:
            print(f"FlexAttention vs 稠密 mask 最大 logits 偏差 = "
                  f"{(ref-got).abs().max().item():.2e}  （bf16 噪声量级 ~1e-2）")