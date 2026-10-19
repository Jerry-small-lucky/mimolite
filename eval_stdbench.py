# -*- coding: utf-8 -*-
"""
标准基准评测（HumanEval / MBPP），与自建评测同口径：
- prompt 模板一致：用户：{question}\n + <|sep|> + 助手：
- 采样参数一致：temp 0.2 / top_k 50 / top_p 0.9 / rep 1.1 / max 512 / seed 2026 / bf16
- pass@k：samples=k，任一通过即通过
HumanEval：question=完整签名+docstring，模型续写函数体，运行 prompt+completion+test。
MBPP：question=问题描述+签名行（签名从参考实现提取首行 def），模型续写，运行签名+completion+test_list。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval_code import load_model, generate_batch, extract_code  # noqa: E402

BAN_IMPORTS = ["subprocess", "os.system", "socket", "requests", "urllib", "shutil", "sys.exit",
               "pickle", "marshal", "ctypes", "importlib", "eval(", "exec(", "open("]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract_code_std(text: str) -> str:
    """优先标准代码块；其次未闭合代码块尾部；最后去掉推理前缀。"""
    code = extract_code(text)
    if code:
        return code
    m = re.findall(r"```(?:python)?\s*\n?(.*)", text, re.S)
    if m:
        return m[-1].rstrip("`").strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("推理"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def run_code(code: str, tests: str, timeout: int) -> tuple[bool, str]:
    for bad in BAN_IMPORTS:
        if bad in code:
            return False, f"banned:{bad}"
    script = code + "\n\n" + tests
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if r.returncode != 0 or "Traceback" in r.stderr:
        tail = (r.stderr or "").strip().splitlines()[-1] if (r.stderr or "").strip() else "rc!=0"
        return False, f"runtime_error|{tail[:200]}"
    return True, "pass"


def load_bench(name: str) -> list[dict]:
    if name == "humaneval":
        t = pq.read_table(ROOT / "data/stdbench/humaneval.parquet").to_pylist()
        # 指令式适配：模型输入=签名+docstring+中文指令；运行时前缀=纯 Python prompt
        items = [{"id": r["task_id"],
                  "question": r["prompt"].rstrip() + f"\n请实现函数 {r['entry_point']}，输出完整代码。",
                  "code_prefix": r["prompt"],
                  "tests": r["test"].rstrip() + f"\ncheck({r['entry_point']})"} for r in t]
    else:  # mbpp
        t = pq.read_table(ROOT / "data/stdbench/mbpp_test.parquet").to_pylist()
        items = []
        for r in t:
            lines = [l for l in r["code"].splitlines()]
            def_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("def ")), 0)
            sig = "\n".join(x.strip() for x in lines[:def_idx + 1])
            fn = sig.split("def ")[-1].split("(")[0].strip() if "def " in sig else ""
            q = f"{r['prompt'].strip()}\n请实现函数 {fn}：\n{sig}"
            tests = "\n".join(r["test_list"])
            items.append({"id": str(r["task_id"]), "question": q, "code_prefix": sig,
                          "tests": tests,
                          "test_imports": " ".join(r.get("test_imports", []))})
    return items


def assemble(prefix: str, code: str) -> str:
    """组装：prefix=合法 Python 前缀（签名/docstring）；code 若以 def 开头则去 def 行防重复。"""
    c = code.lstrip()
    if c.startswith("def "):
        lines = c.splitlines()
        rest = "\n".join(lines[1:]).lstrip("\n")
        return prefix.rstrip() + "\n" + rest
    return prefix.rstrip() + "\n" + c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs/sft_v8/final.pt"))
    ap.add_argument("--bench", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from eval_code import GEN_PARAMS
    GEN_PARAMS["temperature"] = args.temp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta, ck = load_model(args.ckpt, device)
    items = load_bench(args.bench)
    if args.limit:
        items = items[:args.limit]
    log(f"{args.bench} {len(items)} 题 | pass@{args.samples} | ckpt={args.ckpt}")

    passed, fails, per = 0, [], []
    t0 = time.time()
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        qs = [c["question"] for c in chunk]
        ok = [False] * len(chunk)
        for s in range(args.samples):
            out = generate_batch(model, tok, meta, qs, device, args.max_new,
                                 args.seed + s, bf16=(device == "cuda"))
            for i, (text, _) in enumerate(out):
                if ok[i]:
                    continue
                code = extract_code_std(text)
                full = assemble(chunk[i]["code_prefix"], code)
                tests = chunk[i]["tests"]
                if chunk[i].get("test_imports"):
                    tests = chunk[i]["test_imports"] + "\n" + tests
                ok_i, _ = run_code(full, tests, args.timeout)
                if ok_i:
                    ok[i] = True
        for i, c in enumerate(chunk):
            per.append({"id": c["id"], "pass": ok[i]})
            if ok[i]:
                passed += 1
            else:
                fails.append({"id": c["id"], "pass": False})
        if (start + args.batch) % 50 == 0 or start + args.batch >= len(items):
            el = time.time() - t0
            log(f"{min(start+args.batch, len(items))}/{len(items)} 题 | 已通过 {passed} ({el:.0f}s)")

    rate = passed / len(items)
    result = {"bench": args.bench, "ckpt": str(args.ckpt), "n": len(items),
              "n_pass": passed, "pass_rate": round(rate, 4),
              "gen": {"temperature": args.temp, "samples": args.samples,
                      "top_k": GEN_PARAMS.get("top_k"), "top_p": GEN_PARAMS.get("top_p"),
                      "rep": GEN_PARAMS.get("repetition_penalty")}}
    out = args.out or str(ROOT / f"logs/std_{args.bench}_s{args.samples}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": result, "summary": {"n_pass": passed, "pass_rate": rate},
                   "failures": fails, "per_item": per}, f, ensure_ascii=False, indent=1)
    log(f"完成 {args.bench}: pass@{args.samples} = {passed}/{len(items)} = {rate:.1%} -> {out}")


if __name__ == "__main__":
    main()
