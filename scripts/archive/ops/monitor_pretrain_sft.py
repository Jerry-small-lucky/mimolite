# -*- coding: utf-8 -*-
"""流水线监控 v2：等待 pretrain_v2 完成 -> 自动 SFT -> 验证生成 -> 通过则删除旧版错位权重 runs/a001_B1"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\37595\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = Path(r"F:\自研大模型")
PLOG = ROOT / "logs" / "pretrain_v2_stdout.log"
SLOG = ROOT / "logs" / "sft_v2_stdout.log"
SERR = ROOT / "logs" / "sft_v2_stderr.log"
PLOG2 = ROOT / "logs" / "pipeline_v2.log"
FLAG = ROOT / "logs" / "pipeline_v2_done.flag"
OLD_RUN = ROOT / "runs" / "a001_B1"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with PLOG2.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def verify_generation() -> bool:
    """加载 SFT 权重跑 2 题，判定：输出长度达标且无崩溃重复。"""
    sys.path.insert(0, str(ROOT))
    import torch
    from tokenizers import Tokenizer
    from model import MiMoLite, MiMoLiteConfig

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    meta = json.loads((ROOT / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    ck = torch.load(ROOT / "runs/sft/final.pt", map_location="cpu", weights_only=False)
    model = MiMoLite(MiMoLiteConfig(**ck["config"])).to("cuda")
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    bad = re.compile(r"([a-zA-Z0-9])\1{7,}")  # 字母/数字重复 8+ 视为崩溃
    sep = tok.token_to_id("<|sep|>")
    questions = [
        "统计字符串中每个字符出现的次数，并返回一个字典。",
        "用 Python 写一个函数，计算整数列表的平均值。",
    ]
    ok = True
    for i, q in enumerate(questions, 1):
        ids = [meta["bos"]] + tok.encode(f"用户：{q}\n").ids + [sep] + tok.encode("助手：").ids
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=256, temperature=0.6, top_k=50, top_p=0.9,
                                 eos_id=meta["eos"], repetition_penalty=1.1, max_context=1024)
        gen = out[0][len(ids):].tolist()
        if meta["eos"] in gen:
            gen = gen[:gen.index(meta["eos"])]
        text = tok.decode(gen)
        ok_len = len(text) >= 20
        ok_rep = not bool(bad.search(text))
        ok = ok and ok_len and ok_rep
        log(f"验证题{i}: len={len(text)} 重复崩溃={not ok_rep} 通过={ok_len and ok_rep}")
        if not (ok_len and ok_rep):
            log(f"  输出片段: {text[:120]!r}")
    return ok


def main() -> None:
    log("监控 v2 启动，等待预训练完成")
    deadline = time.time() + 135 * 60
    while time.time() < deadline:
        txt = PLOG.read_text(encoding="utf-8", errors="ignore") if PLOG.exists() else ""
        if "训练完成" in txt:
            log("预训练完成，启动 SFT")
            break
        time.sleep(60)
    else:
        log("等待预训练超时，退出")
        sys.exit(1)

    cmd = [PY, str(ROOT / "sft.py"),
           "--ckpt", str(ROOT / "runs/a001_B1_v2/final.pt"),
           "--data", str(ROOT / "data/synth_code_sft.jsonl"),
           "--run-name", "sft",
           "--epochs", "3", "--val-n", "60",
           "--micro-bs", "16", "--accum", "2",
           "--max-len", "512", "--lr", "1e-4",
           "--warmup", "10", "--wd", "0.05", "--clip", "1.0"]
    with SLOG.open("w", encoding="utf-8") as so, SERR.open("w", encoding="utf-8") as se:
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=so, stderr=se)
    log(f"SFT 退出码 {r.returncode}")
    if r.returncode != 0:
        log("SFT 失败，跳过删除，保留旧版 a001_B1")
        sys.exit(1)

    log("开始生成验证")
    ok = verify_generation()
    if ok:
        if OLD_RUN.exists():
            shutil.rmtree(OLD_RUN)
            log("验证通过，已删除旧版错位权重 runs/a001_B1")
        else:
            log("验证通过，旧版 a001_B1 不存在（无需删除）")
    else:
        log("验证未通过，保留旧版 a001_B1 供排查")
    FLAG.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + (" verify_ok" if ok else " verify_fail"), encoding="utf-8")
    log("流水线完成")


if __name__ == "__main__":
    main()
