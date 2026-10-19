"""A-001 守护训练：等 GPU 空闲自动训练，被占用自动暂停续跑。
- 检测 nvidia-smi：显存 <2.5G 且 util <40% 且持续 45s 视为 GPU 空闲
- 每配置 --save-every 100 高频存档；崩溃后等 GPU 空闲 --resume 续跑
- 三配置串行；每配置最多重试 15 次，超限放弃并退出
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\37595\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = Path(r"F:\自研大模型")
RUNS = ROOT / "runs"
LOG = ROOT / "logs" / "a001_watch.log"

# B-001：L8 胜出结构继续预训练（8×960 全注意力，4500 步 ≈ 73.9M tokens 用尽 v1 子池）
BASE = [
    "--steps", "4500", "--seq-len", "512", "--micro-bs", "4", "--accum", "8",
    "--lr", "0.0015", "--sched", "cosine", "--gradpower-p", "1.0",
    "--eval-every", "300", "--save-every", "100", "--seed", "1234",
    "--wsd-decay-ratio", "0.2", "--attn-impl", "dense", "--ga-interval", "1",
    "--zh-data", "data/a001_zh.jsonl", "--en-data", "data/a001_en.jsonl",
]
# A-001 已完成的消融配置（保留存档）
# ("a001_L20", ["--n-layer", "20", "--d-model", "640", "--n-head", "10", "--d-ff", "1728"]),
# ("a001_L14", ["--n-layer", "14", "--d-model", "768", "--n-head", "12", "--d-ff", "2048"]),
# ("a001_L8", ["--n-layer", "8", "--d-model", "960", "--n-head", "15", "--d-ff", "2560"]),
CONFIGS = [
    ("a001_B1", ["--n-layer", "8", "--d-model", "960", "--n-head", "15", "--d-ff", "2560"]),
]
MAX_RETRY = 15


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_free() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        parts = [x.strip() for x in out.split(",")]
        mem, util = int(parts[0]), int(parts[1])
        return mem < 2500 and util < 40
    except Exception:
        return False


def wait_gpu_free(min_secs: int = 45) -> None:
    stable = 0
    while True:
        if gpu_free():
            stable += 15
            log(f"GPU 空闲中（{stable}/{min_secs}s）")
            if stable >= min_secs:
                return
        else:
            if stable > 0:
                log("GPU 被占用，暂停等待")
            stable = 0
        time.sleep(15)


def run_config(name: str, extra: list[str]) -> bool:
    run_dir = RUNS / name
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    retries = 0
    while True:
        wait_gpu_free()
        resume = (run_dir / "last.pt").exists()
        cmd = [PY, "pretrain.py", "--run-name", name] + BASE + extra
        if resume:
            cmd.append("--resume")
        log(f"启动 {name} {'（断点续跑）' if resume else '（新 run）'}")
        p = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if p.returncode == 0:
            log(f"{name} 训练完成（rc=0）")
            return True
        retries += 1
        log(f"{name} 异常退出（rc={p.returncode}），重试 {retries}/{MAX_RETRY}")
        if retries >= MAX_RETRY:
            log(f"{name} 重试超限，放弃")
            return False
        time.sleep(30)


if __name__ == "__main__":
    log("=== 守护训练启动 ===")
    done = []
    for name, extra in CONFIGS:
        if run_config(name, extra):
            done.append(name)
        else:
            break
    log(f"完成配置：{done}")

    # B-001 完成后自动接力：蒸馏合成 v2（可验证代码数据）
    if done and done[-1] == "a001_B1":
        log("B-001 完成，接力启动蒸馏合成 v2（可验证代码数据）")
        retries = 0
        while True:
            wait_gpu_free()
            log("启动 gen_synth_v2.py（蒸馏合成）")
            p = subprocess.run([PY, "scripts/gen_synth_v2.py"], cwd=str(ROOT), env=env)
            if p.returncode == 0:
                log("蒸馏合成完成（rc=0）")
                break
            retries += 1
            log(f"蒸馏异常退出（rc={p.returncode}），重试 {retries}/{MAX_RETRY}")
            if retries >= MAX_RETRY:
                log("蒸馏重试超限，放弃")
                break
            time.sleep(30)
    log("=== 守护结束 ===")
