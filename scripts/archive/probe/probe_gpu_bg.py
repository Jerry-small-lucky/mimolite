"""GPU 后台利用率探针：后台任务持续做 matmul，前台采样 nvidia-smi 判断后台是否满血。"""
import sys, time, json, subprocess
import torch

def smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return "nvidia-smi failed"

print("cuda available:", torch.cuda.is_available(), file=sys.stderr, flush=True)
if not torch.cuda.is_available():
    print("NO_CUDA", flush=True)
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0), file=sys.stderr, flush=True)

a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
t0 = time.time()
n = 0
while time.time() - t0 < 600:
    c = a @ b
    n += 1
    if n % 20 == 0:
        print(f"[probe] iter={n} smi={smi()}", flush=True)
    torch.cuda.synchronize()
print(f"[probe] done iters={n}", flush=True)
