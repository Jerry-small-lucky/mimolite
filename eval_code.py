# -*- coding: utf-8 -*-
"""
代码通过率评测（手册第一步 · 建立评测基线）
====================================================================
流程（与手册 2.3 一致）：
  加载 runs/sft/final.pt，按模板 用户：{题目}\n + <|sep|> + 助手： 构造提示词
  生成：temperature 0.2 / max_new_tokens 512 / top_k 50 / top_p 0.9 / repetition_penalty 1.1
  提取 ```python ... ``` 代码块；无代码块记该题未通过
  沙箱：代码+评测测试用 python 子进程真跑，超时 20s，禁止 os.system/subprocess/eval/exec/
        open/input/socket 等危险调用（复用 gen_synth_v2.py 的 BAN_IMPORTS 黑名单）
  统计：通过率、平均生成长度、失败样例清单；结果落盘 logs/eval_code.json 便于历次对比

用法：
  python eval_code.py --ckpt runs/sft/final.pt --data data/eval_code.jsonl
  python eval_code.py --limit 4          # 冒烟测试：只跑前 4 条
  python eval_code.py --samples 8        # pass@k：每题采样 8 次，任一通过即该题通过
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

# 与 scripts/gen_synth_v2.py 完全一致的安全黑名单（手册要求复用蒸馏脚本黑名单）
BAN_IMPORTS = ("os.system", "subprocess", "eval(", "exec(", "pickle",
               "socket", "requests", "urllib", "open(", "input()", "sys.stdin",
               "shutil", "ctypes", "multiprocessing", "threading")

GEN_PARAMS = dict(temperature=0.2, top_k=50, top_p=0.9, repetition_penalty=1.1,
                  max_context=1024)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_model(ckpt_path: str, device: str):
    from chat import load
    return load(ckpt_path, device)


def build_prompt_ids(tok, meta, question: str) -> list[int]:
    """与 chat.py / 训练模板一致：用户：{题目}\n + <|sep|> + 助手："""
    sep = meta.get("sep") if isinstance(meta.get("sep"), int) else tok.token_to_id("<|sep|>")
    text = f"用户：{question}\n"
    return [meta["bos"]] + tok.encode(text).ids + [sep] + tok.encode("助手：").ids


@torch.no_grad()
def generate_batch(model, tok, meta, questions: list[str], device: str,
                   max_new: int, seed_base: int, bf16: bool = True) -> list[tuple[str, int]]:
    """批量自回归生成；返回 [(解码文本, 生成token数)]。逐 batch 固定种子保证可复现。
    bf16 推理：与训练（bf16 autocast）口径一致，实测提速 ~4.5×。"""
    pad = meta["pad"]
    idss = [build_prompt_ids(tok, meta, q) for q in questions]
    max_len = max(len(x) for x in idss)
    x = torch.full((len(idss), max_len), pad, dtype=torch.long)
    for i, ids in enumerate(idss):
        x[i, :len(ids)] = torch.tensor(ids)
    x = x.to(device)

    # 每个 batch 一个确定性种子（基准可复现；不受并行度影响）
    torch.manual_seed(seed_base)
    torch.cuda.manual_seed_all(seed_base)
    gen = torch.Generator(device="cuda" if device == "cuda" else "cpu")
    gen.manual_seed(seed_base)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(bf16 and device == "cuda")):
        out = model.generate(x, max_new_tokens=max_new, eos_id=meta["eos"],
                             generator=gen, **GEN_PARAMS)
    results = []
    for i, ids in enumerate(idss):
        row = out[i][len(ids):].tolist()
        if meta["eos"] in row:
            row = row[:row.index(meta["eos"])]
        results.append((tok.decode(row), len(row)))
    return results


def extract_code(text: str) -> str | None:
    """提取 ```python ... ``` 代码块（与 gen_synth_v2.extract_blocks 同正则）。"""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return blocks[0].strip() if blocks else None


def run_sandbox(code: str, tests: str, timeout: int) -> tuple[bool, str]:
    """沙箱执行 code+tests：黑名单检查 + 子进程真跑。返回 (通过?, 原因)。"""
    for bad in BAN_IMPORTS:
        if bad in code:
            return False, f"banned:{bad}"
    script = code + "\n\n" + tests
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if r.returncode != 0 or "Traceback" in r.stderr:
        tail = (r.stderr or "").strip().splitlines()[-1] if (r.stderr or "").strip() else "rc!=0"
        return False, f"runtime_error|{tail[:200]}"
    return True, "pass"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/sft/final.pt"))
    ap.add_argument("--data", type=str, default=str(ROOT / "data/eval_code.jsonl"))
    ap.add_argument("--out", type=str, default=None,
                    help="结果路径；默认 logs/eval_code.json（pass@1）或 logs/eval_code_p{k}.json")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--samples", type=int, default=1,
                    help="pass@k：每题采样次数 k（默认 1=pass@1），任一通过即该题通过")
    ap.add_argument("--no-bf16", action="store_true", help="关闭 bf16 加速（默认开启）")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 条（冒烟测试）")
    ap.add_argument("--sig", action="store_true",
                    help="prompt 附函数签名（从 tests 提取函数名，对齐函数名能力）")
    args = ap.parse_args()

    if not args.out:
        args.out = str(ROOT / ("logs/eval_code.json" if args.samples == 1
                               else f"logs/eval_code_p{args.samples}.json"))

    GEN_PARAMS["temperature"] = args.temp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta, ck = load_model(args.ckpt, device)
    log(f"载入 {args.ckpt} (step={ck.get('step')}), "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M, device={device}")

    # ---- 数据 ----
    items = [json.loads(l) for l in Path(args.data).open(encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[:args.limit]

    def prompt_of(item: dict) -> str:
        q = item["question"]
        if args.sig:
            m = re.search(r"(\w+)\s*\(", item["tests"])
            if m:
                fn = m.group(1)
                q = f"{q}\n请实现函数 {fn}。" if item.get("lang") != "en" else f"{q}\nPlease implement function {fn}."
        return q

    log(f"评测集 {len(items)} 条 <- {args.data} | pass@k k={args.samples} "
        f"(共 {len(items) * args.samples} 次生成) | sig={args.sig}")

    # 平铺采样任务：(item_idx, sample_idx)
    tasks = [(i, s) for i in range(len(items)) for s in range(args.samples)]
    sb_cache: dict[tuple[str, str], tuple[bool, str]] = {}   # (code, tests) -> 结果，去重沙箱
    _acc: dict[int, list[dict]] = {}        # item_idx -> 各采样结果（收集到 samples 条后聚合）
    _done_ids: dict[int, bool] = {}         # 已聚合进 per_item 的题目

    t0 = time.time()
    per_item: list[dict] = []
    failures: list[dict] = []
    n_pass = 0
    batch_counter = 0

    def save_partial(done: bool = False):
        dur = time.time() - t0
        n = len(per_item)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "meta": {
                "ckpt": args.ckpt, "data": args.data, "n_total": len(items),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gen": {**GEN_PARAMS, "max_new_tokens": args.max_new, "batch": args.batch,
                        "seed": args.seed, "samples": args.samples, "timeout_s": args.timeout,
                        "bf16": not args.no_bf16},
                "device": device, "done": done,
            },
            "summary": {
                "n": n, "n_pass": n_pass, "n_fail": n - n_pass,
                "pass_rate": round(n_pass / max(1, n), 4),
                "avg_gen_tokens": round(sum(i["gen_tokens"] for i in per_item) / max(1, n), 1),
                "avg_gen_chars": round(sum(i["gen_chars"] for i in per_item) / max(1, n), 1),
                "duration_s": round(dur, 1),
            },
            "failures": failures,
            "per_item": per_item,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        for start in range(0, len(tasks), args.batch):
            chunk = tasks[start:start + args.batch]
            qs = [prompt_of(items[i]) for i, _ in chunk]
            # pass@1 保持历史行为（每 batch 固定 seed）；pass@k 每个 batch 递增 seed 保证采样多样性
            seed = args.seed if args.samples == 1 else args.seed + batch_counter
            tb = time.time()
            texts_lens = generate_batch(model, tok, meta, qs, device,
                                        args.max_new, seed, bf16=not args.no_bf16)
            tg = time.time() - tb
            for (i, s), (text, ntok) in zip(chunk, texts_lens):
                code = extract_code(text)
                ok, reason = False, "no_code_block"
                if code is not None:
                    key = (code, items[i]["tests"])
                    if key not in sb_cache:
                        sb_cache[key] = run_sandbox(code, items[i]["tests"], args.timeout)
                    ok, reason = sb_cache[key]
                sample_rec = {"pass": ok, "reason": reason, "gen_tokens": ntok,
                              "gen_chars": len(text), "sample": s,
                              "gen_excerpt": text[:200] if not ok else ""}
                if i not in _acc:
                    _acc[i] = []
                _acc[i].append(sample_rec)
            batch_counter += 1

            # 每当一个题目的全部采样完成，聚合写入 per_item
            done_n = 0
            for i in sorted(_acc):
                if len(_acc[i]) == args.samples and i not in _done_ids:
                    recs = _acc[i]
                    n_ok = sum(1 for r in recs if r["pass"])
                    ok = n_ok > 0
                    if ok:
                        n_pass += 1
                    first = recs[0]
                    per_item.append({
                        "id": items[i]["id"], "pass": ok,
                        "n_pass_samples": n_ok, "samples": len(recs),
                        "reason": first["reason"],
                        "gen_tokens": round(sum(r["gen_tokens"] for r in recs) / len(recs), 1),
                        "gen_chars": round(sum(r["gen_chars"] for r in recs) / len(recs), 1),
                    })
                    if not ok:
                        failures.append({
                            "id": items[i]["id"], "pass": False,
                            "n_pass_samples": n_ok, "samples": len(recs),
                            "reason": first["reason"],
                            "question": items[i]["question"][:80],
                            "gen_excerpt": first["gen_excerpt"][:200],
                        })
                    _done_ids[i] = True
                    done_n = len(per_item)

            if done_n and (done_n % 10 == 0 or done_n == len(items)):
                el = time.time() - t0
                eta = el / done_n * (len(items) - done_n)
                log(f"{done_n}/{len(items)} 题 | 通过 {n_pass} ({n_pass/done_n*100:.0f}%) "
                    f"| {el:.0f}s 已用, 预计剩余 {eta:.0f}s")
                save_partial(done=(done_n == len(items)))
    finally:
        save_partial(done=(len(per_item) == len(items)))

    dur = time.time() - t0
    log(f"完成：{len(per_item)} 题（k={args.samples}），通过 {n_pass}"
        f"（{n_pass/max(1,len(per_item))*100:.1f}%），"
        f"平均每题生成 {sum(i['gen_tokens'] for i in per_item)/max(1,len(per_item)):.0f} tokens，"
        f"用时 {dur:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
