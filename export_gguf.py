"""
导出 GGUF —— 让 MiMo-Lite 直接跑在你本机的 llama.cpp 上
=======================================================

为什么用 `qwen3` 架构导出？
  MiMo-Lite 的骨干（RMSNorm + RoPE + SwiGLU + QK-Norm + 无 bias + GQA）
  与 Qwen3 完全同构，而 llama.cpp 对 qwen3 架构支持最完整，
  并且支持滑动窗口注意力 —— 正好承载小米 MiMo 的混合注意力设计。

MTP 模块不会被导出：llama.cpp 目前不认识 MTP 头。
它只在预训练阶段提供辅助监督信号，推理阶段由主模型承担。
（投机解码草稿模型是后续可玩的方向。）

用法：
  python export_gguf.py --ckpt runs/pretrain/final.pt --out mimolite-base
  python export_gguf.py --ckpt ... --out ... --attn-mode full   # 关闭 SWA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# PyTorch 权重名 -> GGUF 张量名
TENSOR_MAP = {
    "embed.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
LAYER_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.q_proj.weight": "attn_q.weight",
    "attn.k_proj.weight": "attn_k.weight",
    "attn.v_proj.weight": "attn_v.weight",
    "attn.o_proj.weight": "attn_output.weight",
    "attn.q_norm.weight": "attn_q_norm.weight",
    "attn.k_norm.weight": "attn_k_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn.gate_proj.weight": "ffn_gate.weight",
    "ffn.up_proj.weight": "ffn_up.weight",
    "ffn.down_proj.weight": "ffn_down.weight",
}


def load_tokenizer_vocab(tok_path: Path):
    """
    从 tokenizer.json 取出 vocab 与 merges（byte-level BPE）。

    注意：不能用 `Tokenizer.from_file(...).model.merges`，
    本机 tokenizers 版本加载后该字段为空，必须直接读 JSON。
    """
    raw = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab = raw["model"]["vocab"]
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t

    merges = []
    for m in raw["model"].get("merges", []):
        if isinstance(m, str):
            merges.append(m)
        else:
            merges.append(f"{m[0]} {m[1]}")

    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    return tokens, merges, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/pretrain/final.pt"))
    ap.add_argument("--out", type=str, default=str(ROOT / "out/mimolite-base"))
    ap.add_argument("--attn-mode", choices=["swa", "full"], default="swa",
                    help="swa=保留滑动窗口元数据；full=全注意力")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--name", type=str, default="MiMo-Lite")
    args = ap.parse_args()

    import gguf

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["model"]
    meta = ckpt.get("tokenizer_meta") or json.loads(
        (ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))

    tokens, merges, _ = load_tokenizer_vocab(ROOT / "tokenizer" / "tokenizer.json")
    log(f"tokenizer: {len(tokens)} tokens, {len(merges)} merges")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".gguf")

    gw = gguf.GGUFWriter(str(out_path), "qwen3")

    # ---- 超参数 ----
    gw.add_name(args.name)
    gw.add_context_length(args.ctx)
    gw.add_embedding_length(cfg["d_model"])
    gw.add_block_count(cfg["n_layer"])
    gw.add_feed_forward_length(cfg["d_ff"])
    gw.add_rope_dimension_count(cfg["d_head"])
    gw.add_rope_freq_base(cfg["rope_theta"])
    gw.add_head_count(cfg["n_head"])
    gw.add_head_count_kv(cfg["n_kv_head"])
    gw.add_layer_norm_rms_eps(cfg["norm_eps"])
    gw.add_key_length(cfg["d_head"])
    gw.add_value_length(cfg["d_head"])
    gw.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    if args.attn_mode == "swa":
        from model import ga_layer_indices
        gw.add_sliding_window(cfg["window"])
        ga = ga_layer_indices(cfg)
        gset = set(ga)
        swa = [i for i in range(int(cfg["n_layer"])) if i not in gset]
        # 记录非连续的混合注意力层号（SWA 交错/分块，而非"前 n 层"式连续窗口）
        if swa:
            gw.add_key_value("qwen3.swa_nums", swa,
                             gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        if ga:
            gw.add_key_value("qwen3.ga_nums", ga,
                             gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        log(f"滑动窗口={cfg['window']}, SWA 层={swa}, GA 层={ga}")
    else:
        log("全注意力模式（不写滑动窗口元数据）")

    # ---- tokenizer ----
    gw.add_tokenizer_model("gpt2")
    special = {meta["pad"], meta["bos"], meta["eos"], meta.get("sep", 3)}
    tok_types = [
        gguf.TokenType.CONTROL if i in special else gguf.TokenType.NORMAL
        for i in range(len(tokens))
    ]
    # 预分词器必须是 "gpt-2"：本 tokenizer 用的是 HF ByteLevel（GPT-2 同款正则）。
    # 写成 "default" 会让 llama.cpp 在词首/空白处的切分与训练时不一致，
    # 表现为全文 token 数对不上（实测 1110 vs 1104），进而困惑度和生成结果都偏。
    gw.add_tokenizer_pre("gpt-2")
    gw.add_token_list(tokens)
    gw.add_token_scores([-float(i) if i not in special else 0.0
                         for i in range(len(tokens))])
    gw.add_token_types(tok_types)
    gw.add_token_merges(merges)
    gw.add_bos_token_id(meta["bos"])
    gw.add_eos_token_id(meta["eos"])
    # llama.cpp 靠 eot_token_id 构建 special_eog_ids；不写会一直告警
    # "special_eos_id is not in special_eog_ids"，且生成时无法正确判停。
    gw.add_eot_token_id(meta["eos"])
    gw.add_pad_token_id(meta["pad"])
    gw.add_unk_token_id(meta["pad"])
    gw.add_add_bos_token(True)
    gw.add_add_eos_token(False)

    # ---- 权重 ----
    # dtype 策略（对齐 llama.cpp 社区惯例）：
    #   - 2D 权重矩阵 -> F16，体积减半，且与 file_type 声明一致
    #   - 1D 归一化权重 -> 保留 F32。RMSNorm 权重对数值精度敏感，
    #     且它们只占全部参数的极小部分，没有省体积的必要
    written, skipped = 0, 0
    for name, tensor in sd.items():
        if name.startswith("mtp."):
            skipped += 1
            continue
        if name in TENSOR_MAP:
            tname = TENSOR_MAP[name]
        elif name.startswith("layers."):
            parts = name.split(".")
            idx = parts[1]
            suffix = ".".join(parts[2:])
            if suffix not in LAYER_MAP:
                skipped += 1
                continue
            tname = f"blk.{idx}.{LAYER_MAP[suffix]}"
        else:
            skipped += 1
            continue

        t = tensor.detach()
        if t.ndim == 1:
            arr = t.float().numpy().astype(np.float32)
        else:
            # F16 上限 65504，超出则保留 F32，避免 inf
            if float(t.abs().max()) > 60000.0:
                arr = t.float().numpy().astype(np.float32)
            else:
                arr = t.half().float().numpy().astype(np.float16)
        gw.add_tensor(tname, arr)
        written += 1

    log(f"写入 {written} 个张量，跳过 {skipped} 个（MTP 等）")
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    size_mb = out_path.stat().st_size / 1e6
    log(f"导出完成 -> {out_path} ({size_mb:.1f} MB)")

    info = out_path.with_suffix(".json")
    info.write_text(json.dumps({
        "name": args.name,
        "arch": "qwen3",
        "config": cfg,
        "attn_mode": args.attn_mode,
        "step": ckpt.get("step"),
        "params_M": round(sum(v.numel() for k, v in sd.items()
                              if not k.startswith("mtp.")) / 1e6, 2),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"配置信息 -> {info}")


if __name__ == "__main__":
    main()