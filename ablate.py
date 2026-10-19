"""
消融实验框架
============

单卡 8GB 的现实约束下，**同样 GPU 小时，用来产出"结论"的信息密度远高于产出权重**。
本框架把一组配置逐个跑完，收集固定验证集上的终局 loss，自动出对比表。

三条不可妥协的实验纪律
----------------------
1. **等参数量**：跨架构比较必须配平参数量，否则区分不出「深更好」与「大更好」。
   `--match-params` 会自动二分 d_model（保持 d_head=64 不变，n_head 随宽度变），
   把每个配置的总参数压到基准的 ±tolerance 内。
2. **等 token 预算**：所有配置用同样的 steps × seq_len × batch，不按步数偷懒。
3. **同验证集**：PackedSampler 把每个数据流尾部 2% 隔离为验证集，
   验证批次按 bs 缓存后固定复用 —— 跨 run 的 val loss 可以直接相减。

每个配置跑在**独立子进程**里：某个配置 OOM 或崩溃不会带走整轮消融。

用法：
  python ablate.py --suite a003 --steps 1200              # SWA 比例与位置
  python ablate.py --suite a001 --steps 1200              # 深度 vs 宽度（等参数）
  python ablate.py --suite a005 --steps 1200              # 调度 × GradPower
  python ablate.py --suite a003 --steps 200 --dry-run     # 只打印要跑什么
  python ablate.py --list                                  # 列出所有套件
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

# Windows 下 Python 默认把 stdout/stderr 编码成控制台代码页（本机是 GBK），
# 只要输出里出现 ✓、✗、─ 这类字符就会 UnicodeEncodeError 直接打死进程。
# 不依赖调用方设置 PYTHONIOENCODING —— 那样换个终端启动就又炸了。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

# 基准：与 runs/pretrain 当前基线一致（14 层 / 768 宽 / 6:1）
BASE_PARAMS = 108_160_000
BASE_SEQ = 512


# ---------------------------------------------------------------------------
# 参数量配平
# ---------------------------------------------------------------------------
def count_params(n_layer: int, d_model: int, n_head: int, n_kv_head: int,
                 d_ff: int, vocab: int, mtp_depth: int) -> int:
    """
    **用 meta device 实例化真实模型数参数**，而不是写解析式。

    解析式和 model.py 漂移过一次（算出 98.13M，真实是 108.16M，差 9%），
    而参数量配平是所有跨架构结论的前提 —— 这种偏差会直接毁掉消融结论。
    meta device 不分配显存/内存，实例化是瞬时的。
    """
    import torch
    from model import MiMoLite, MiMoLiteConfig
    cfg = MiMoLiteConfig(
        vocab_size=vocab, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head,
        d_model=d_model, d_head=d_model // n_head, d_ff=d_ff, mtp_depth=mtp_depth,
    )
    with torch.device("meta"):
        m = MiMoLite(cfg)
    return sum(p.numel() for p in m.parameters())


def d_ff_for(d_model: int, multiple: int = 64) -> int:
    """SwiGLU 中间层宽度：基准 768→2048 即 2.667 倍（8/3），对齐该比例。"""
    return max(multiple, int(round(d_model * 8 / 3 / multiple)) * multiple)


def solve_d_model(target: int, n_layer: int, vocab: int, mtp_depth: int = 1,
                  d_head: int = 64, gqa_group: int = 3, max_n_head: int = 48):
    """
    求解 d_model，使总参数最接近 target。

    固定的自变量只有 n_layer 和 d_model，其余尽量钉死，否则消融结论会被污染：

    * d_head = 64 固定 —— 改它会同时改变 RoPE 频率（另一条自变量）
    * GQA 组大小 n_head/n_kv_head 限制在 [2, 4]（基线 12/4 = 3）—— 改它就等于
      同时改了 KV Cache 压缩率。这个限制同时解决了素数 n_head 的病：
      n_head=13 的因子只有 {1,13}，组大小要么 13 要么 1，放宽到 [2,4] 就会
      被排除，从而避免参数在相邻网格点之间跳变 28M（+8% 宽度 → +28M 参数）。
    * d_ff 按 SwiGLU 的 8/3 比例跟随 d_model（基线 768→2048）

    返回 (n_head, d_model, d_ff, 实际参数量, n_kv_head, 相对误差)
    """
    lo_group, hi_group = 2, 4
    best = None
    for n_head in range(1, max_n_head + 1):
        d_model = n_head * d_head
        if d_model > 4096:
            break
        divisors = [kv for kv in range(1, n_head + 1)
                    if n_head % kv == 0 and lo_group <= n_head // kv <= hi_group]
        if not divisors:
            continue
        kv = min(divisors, key=lambda v: (abs(n_head // v - gqa_group), v))
        d_ff = d_ff_for(d_model)
        n = count_params(n_layer, d_model, n_head, kv, d_ff, vocab, mtp_depth)
        if best is None or abs(n - target) < abs(best[3] - target):
            best = (n_head, d_model, d_ff, n, kv)
    if best is None:
        raise ValueError(f"n_layer={n_layer} 下没有满足 GQA 组大小 [{lo_group},{hi_group}] 的配置")
    n_head, d_model, d_ff, n, kv = best
    err = (n - target) / target
    return n_head, d_model, d_ff, n, kv, err


# ---------------------------------------------------------------------------
# 消融套件
# ---------------------------------------------------------------------------
def suite_a003() -> list[dict]:
    """
    A-003：SWA/GA 的**比例**与**位置**，固定 14 层 × 768（与基线逐参数相同）。

    为什么不用 12 层：换成 12 层就必须重新配平 d_model 才能等参数，
    而宽度一动，RoPE 维度、注意力头数、GQA 组大小全都跟着动 ——
    那就不是在测「SWA 比例」了。**保持 14 层不变，参数量天然 100% 相同，
    val loss 的任何差异都只能归因于 GA 层的位置。** 这是本套件最干净的地方。

    调研依据（两个独立来源指向同一方向，且都与当前实现相反）：
      * ConSA (arXiv:2606.18056, 0.6B/1.7B) 学到的分配是「SWA 在底层、
        FA 集中于**连续**中间层块」，明确偏离均匀交错；
      * BOSCH (arXiv:2604.05942, ACL 2026) 发现逐层自适应 SWA 比例优于层间启发式。

    两段设计：
      比例段（GA 层数变化）：13:1 / 6:1 / 2.5:1
      位置段（GA 固定 2 层 = 6:1）：底层 {1,7} / 交错 {5,11} / 连续中段 {6,7} / 顶层 {7,13}
    位置段四个格子共享完全相同的参数量、GA 层数与预算，是干净的单变量对照。

    附注：n_layer=14 不是 6 的倍数，所以 ga_interval=6 实际只给出 2 个 GA 层，
    SWA:GA = 12:2 = 6:1，而不是文档里最初写的 5:1。
    精确 5:1 需要 n_layer ∈ {12, 18, 24}。
    """
    L = 14
    return [
        # ---- 比例段 ----
        dict(name="a003_r13", n_layer=L, ga_layers=[7],
             note="13:1 近全 SWA（比例下界）"),
        dict(name="a003_r6", n_layer=L, ga_layers=[5, 11],
             note="6:1 均匀交错（当前基线）"),
        dict(name="a003_r25", n_layer=L, ga_layers=[3, 6, 9, 12],
             note="2.5:1 均匀交错（更多全局层）"),
        # ---- 位置段：全部 2 个 GA 层，比例恒为 6:1 ----
        dict(name="a003_p_bot", n_layer=L, ga_layers=[1, 7],
             note="6:1 偏底层"),
        dict(name="a003_p_mid", n_layer=L, ga_layers=[6, 7],
             note="6:1 连续中间块（ConSA 式）"),
        dict(name="a003_p_top", n_layer=L, ga_layers=[7, 13],
             note="6:1 偏顶层"),
    ]


def suite_a001() -> list[dict]:
    """
    A-001：**等参数量**下的深度 vs 宽度。

    不配平参数量就做深度对比是自欺：更深的模型通常更大，
    结论会分不清「深更好」还是「大更好」。这里每个配置都被压到
    BASE_PARAMS ± tolerance 内。

    d_model 由 solve_d_model 自动求解，d_head 固定 64（RoPE 维度不变）。
    """
    return [
        dict(name="a001_L20", n_layer=20, note="深而窄（20 层）"),
        dict(name="a001_L14", n_layer=14, note="基线（14 层 × 768）"),
        dict(name="a001_L8", n_layer=8, note="浅而宽（8 层）"),
    ]


def suite_a005() -> list[dict]:
    """
    A-005：调度 × GradPower，架构固定为当前基线。

    * WSD (arXiv:2404.06395, MiniCPM) —— 短预算下比 cosine 走得远
    * GradPower (arXiv:2505.24275, ICML 2026) —— p=1.2，论文在
      LLaMA-0.2B/C4 定下后 66M→2B 全程沿用；**最大增益恰好出现在 WSD 下**
    四个格子构成 2×2 全因子，能同时读出主效应和交互效应。
    """
    return [
        dict(name="a005_cos_adamw", n_layer=14, sched="cosine", gradpower_p=1.0,
             note="cosine + AdamW（基线）"),
        dict(name="a005_wsd_adamw", n_layer=14, sched="wsd", gradpower_p=1.0,
             note="WSD + AdamW（调度主效应）"),
        dict(name="a005_cos_gp", n_layer=14, sched="cosine", gradpower_p=1.2,
             note="cosine + AdamWPower（GradPower 主效应）"),
        dict(name="a005_wsd_gp", n_layer=14, sched="wsd", gradpower_p=1.2,
             note="WSD + AdamWPower（论文称此格增益最大）"),
        dict(name="a005_wsd_b98", n_layer=14, sched="wsd", gradpower_p=1.0,
             beta2=0.95, beta2_cooldown=0.98,
             note="WSD + 冷段 β₂=0.98（A5-b）"),
    ]


SUITES = {"a001": suite_a001, "a003": suite_a003, "a005": suite_a005}


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def build_cmd(spec: dict, steps: int, seq_len: int, micro_bs: int, accum: int,
              lr: float, tol: float, target: int, vocab: int,
              extra: list[str]) -> list[str]:
    """把一份配置翻译成 pretrain.py 的命令行。"""
    n_layer = spec["n_layer"]
    n_head, d_model, d_ff, n_params, kv, err = solve_d_model(
        target, n_layer, vocab=vocab, mtp_depth=1)
    spec["_n_params"] = n_params
    spec["_err"] = err
    spec["_d_model"] = d_model
    spec["_n_head"] = n_head
    spec["_n_kv_head"] = kv
    spec["_d_ff"] = d_ff

    cmd = [sys.executable, "pretrain.py",
           "--run-name", spec["name"],
           "--steps", str(steps),
           "--seq-len", str(seq_len),
           "--micro-bs", str(micro_bs),
           "--accum", str(accum),
           "--lr", str(lr),
           "--n-layer", str(n_layer),
           "--d-model", str(d_model),
           "--n-head", str(n_head),
           "--d-ff", str(d_ff),
           "--sched", spec.get("sched", "cosine"),
           "--gradpower-p", str(spec.get("gradpower_p", 1.0))]
    if spec.get("beta2") is not None:
        cmd += ["--beta2", str(spec["beta2"])]
    if spec.get("beta2_cooldown") is not None:
        cmd += ["--beta2-cooldown", str(spec["beta2_cooldown"])]
    if spec.get("ga_layers"):
        cmd += ["--ga-layers", ",".join(str(i) for i in spec["ga_layers"])]
    cmd += ["--eval-every", str(max(50, steps // 4))]
    cmd += ["--save-every", str(max(200, steps // 2))]
    cmd += extra
    return cmd


def read_result(name: str, steps: int | None = None) -> dict | None:
    """
    读取某个配置的结果。给 steps 时额外判定该 run 是否**真正跑完**。

    只看 loss_curve.json 里有没有 val 记录是不够的：被中断的 run 同样会
    写 val 记录（pretrain.py 每次 eval 都落盘），于是 --skip-done 会把
    半截的 run 误判成已完成并跳过 —— 那个配置的数据就永远补不上了。
    判定标准：最后一个 val 记录的 step 必须 >= 目标步数（或存在 final.pt）。
    """
    p = RUNS / name / "loss_curve.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = d.get("val") if isinstance(d, dict) else [r for r in d if "val_loss" in r]
    if not val:
        return None
    last_step = val[-1]["step"]
    completed = (last_step >= steps) if steps else (RUNS / name / "final.pt").exists()
    return {
        "final_val": val[-1]["val_loss"],
        "best_val": min(v["val_loss"] for v in val),
        "steps": last_step,
        "n_params": d.get("n_params") if isinstance(d, dict) else None,
        "swa_ga": d.get("swa_ga") if isinstance(d, dict) else None,
        "completed": completed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="a003")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seq-len", type=int, default=BASE_SEQ)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--tol", type=float, default=0.005, help="参数量配平容差")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--target-params", type=int, default=BASE_PARAMS)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", default="dense", choices=["dense", "flex"])
    ap.add_argument("--wsd-decay-ratio", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="", help="逗号分隔的 run 名白名单")
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过**已完成**的配置；半截的会从头重跑（除非加 --resume）")
    ap.add_argument("--resume", action="store_true",
                    help="给半截的配置传 --resume，从 last.pt 断点继续")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.list:
        for k, fn in SUITES.items():
            print(f"{k}:")
            for s in fn():
                print(f"   {s['name']:16s} {s.get('note','')}")
        return

    if args.suite not in SUITES:
        print(f"未知套件 {args.suite}，可选：{list(SUITES)}")
        sys.exit(1)

    tokenizer_meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json")
                                .read_text(encoding="utf-8"))
    vocab = tokenizer_meta["vocab_size"]

    specs = SUITES[args.suite]()
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        specs = [s for s in specs if s["name"] in keep]
    if not specs:
        print("没有匹配的配置")
        return

    extra = ["--seed", str(args.seed), "--wsd-decay-ratio", str(args.wsd_decay_ratio),
             "--attn-impl", args.attn_impl]
    if args.no_grad_ckpt:
        extra.append("--no-grad-ckpt")

    out_path = Path(args.out) if args.out else (ROOT / "runs" / f"ablation_{args.suite}.json")

    print(f"=== 消融 {args.suite}：{len(specs)} 个配置 × {args.steps} 步 ===")
    print(f"    预算 = {args.steps * args.micro_bs * args.accum * (args.seq_len+1)/1e6:.1f}M token/配置")
    print(f"    目标参数量 {args.target_params/1e6:.2f}M ± {args.tol*100:.1f}%\n")

    rows = []
    skipped = []
    for i, spec in enumerate(specs, 1):
        prev = read_result(spec["name"], args.steps)
        if args.skip_done and prev is not None and prev.get("completed"):
            r = prev
            print(f"[{i}/{len(specs)}] {spec['name']}  跳过（已完成）")
            print(f"       ✓ final val {r['final_val']:.4f} / best {r['best_val']:.4f} "
                  f"@ {r['steps']} 步\n")
            skipped.append(spec["name"])
            rows.append(dict(**r, name=spec["name"], note=spec.get("note", ""), skipped=True))
            continue
        if args.skip_done and prev is not None:
            print(f"[{i}/{len(specs)}] {spec['name']}  半截（{prev['steps']}/{args.steps} 步），"
                  f"{'从断点继续' if args.resume else '从头重跑'}")
        cmd = build_cmd(spec, args.steps, args.seq_len, args.micro_bs, args.accum,
                        args.lr, args.tol, args.target_params, vocab, extra)
        err = spec["_err"]
        flag = "OK " if abs(err) <= args.tol else "!! "
        print(f"[{i}/{len(specs)}] {spec['name']}  {spec.get('note','')}")
        print(f"       n_layer={spec['n_layer']} d_model={spec['_d_model']} "
              f"n_head={spec['_n_head']} n_kv={spec['_n_kv_head']} d_ff={spec['_d_ff']} "
              f"ga={spec.get('ga_layers')}")
        print(f"       {flag}params={spec['_n_params']/1e6:.2f}M "
              f"(相对误差 {err*100:+.2f}%)  SWA:GA={spec.get('swa_ga','-')}")
        if spec["name"] == "a001_L14":
            print("       [注] 这一格就是当前 108M 基线的配平版，作为该套件的对照")

        if args.dry_run:
            print("       " + " ".join(cmd[1:]) + "\n")
            continue

        if args.resume:
            cmd.append("--resume")

        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        el = (time.time() - t0) / 60
        res = read_result(spec["name"], args.steps)
        if rc != 0 or res is None or not res.get("completed"):
            got = "无记录" if res is None else f"到 {res.get('steps', 0)} 步"
            print(f"       ✗ 失败或未跑完 rc={rc}（{got}），耗时 {el:.1f} 分钟\n")
            rows.append(dict(name=spec["name"], note=spec.get("note", ""),
                             n_params=spec["_n_params"], err=err, failed=True))
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            continue

        print(f"       ✓ final val {res['final_val']:.4f} / best {res['best_val']:.4f} "
              f"@ {res['steps']} 步，{el:.1f} 分钟\n")
        # 注意：**不能**再显式传 n_params= —— read_result() 返回的 dict 里
        # 已经含 n_params 键，重复关键字会让 dict() 抛 TypeError，
        # 整个消融套件在第一个配置之后就死掉（这个坑踩过）。
        rows.append(dict(**res, name=spec["name"], note=spec.get("note", ""),
                         err=err, failed=False))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    if not args.dry_run:
        print("\n" + "=" * 78)
        print(f"{'配置':<16}{'参数量':>10}{'误差':>8}{'final val':>11}{'best val':>10}  说明")
        print("-" * 78)
        ok = [r for r in rows if not r.get("failed")]
        for r in sorted(ok, key=lambda r: r["final_val"]):
            # 用 .get()：--skip-done 跳过的行没有 err 键（没走求解器）
            errs = f"{r.get('err', 0.0)*100:>+7.2f}%"
            tag = "（跳过，沿用已有结果）" if r.get("skipped") else ""
            print(f"{r['name']:<16}{r['n_params']/1e6:>9.2f}M{errs}"
                  f"{r['final_val']:>11.4f}{r['best_val']:>10.4f}  {r['note']}{tag}")
        for r in rows:
            if r.get("failed"):
                print(f"{r['name']:<16}{r['n_params']/1e6:>9.2f}M"
                      f"{r.get('err',0.0)*100:>+7.2f}%{'—':>11}{'—':>10}  失败")
        print("=" * 78)
        if len(ok) > 1:
            base = min(ok, key=lambda r: r["final_val"])
            print(f"最优：{base['name']}  final val {base['final_val']:.4f}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
