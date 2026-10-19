"""llama.cpp 推理封装：输出写文件，stdin 接 DEVNULL 避免交互阻塞。"""
import subprocess, sys, os

EXE = r"F:\AI\tools\llama.cpp\llama-cli.exe"
model = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "test"
ntok = sys.argv[3] if len(sys.argv) > 3 else "20"
temp = sys.argv[4] if len(sys.argv) > 4 else "0"
outpath = sys.argv[5] if len(sys.argv) > 5 else r"F:\自研大模型\logs\llama_out.txt"

cmd = [EXE, "-m", model, "-p", prompt, "-n", ntok, "--temp", temp,
       "--seed", "1", "-ngl", "99", "--no-display-prompt"]

with open(outpath, "w", encoding="utf-8") as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                   stdin=subprocess.DEVNULL, timeout=900)

text = open(outpath, encoding="utf-8", errors="replace").read()
lines = text.splitlines()
gen = []
capture = False
for ln in lines:
    if ln.startswith("> "):
        if capture:
            break
        capture = True
        body = ln[2:]
        if body.strip():
            gen.append(body)
        continue
    if capture:
        if ln.startswith(">"):
            break
        gen.append(ln)
result = "\n".join(gen).strip()
print("RESULT_REPR:", repr(result))
print("SAVED:", outpath, os.path.getsize(outpath), "bytes")