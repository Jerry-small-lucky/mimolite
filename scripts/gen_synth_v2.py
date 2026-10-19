"""
蒸馏合成 v2 —— 可验证代码推理数据
====================================================================
教师：ollama qwen3-1.7b（推理模型，输出 <think> 思维链）
流程：题目(种子主题) -> 教师生成 题目+思维链+Python代码+断言测试
      -> 本地沙箱执行 code+tests，全部通过才保留（可验证原则）
      -> 只留「真跑通测试」的样本；失败重试（同题最多 2 次）

产出：
  data/synth_code.jsonl     原始结构 {id, src, lang, topic, question, reasoning, code, tests, verified}
  data/synth_code_sft.jsonl 转 messages 格式（user=题目, assistant=推理+代码），供 sft.py
断点续跑：progress 文件记录游标/统计；输出增量写盘。
单线程串行请求 ollama（避免与训练抢 GPU/双写）。
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3-1.7b"
N_TARGET = 600          # 目标：通过验证的样本数（用户拍板 600）
MAX_RETRY_PER_TOPIC = 2  # 每题失败重试次数
PROG = DATA / "synth_code_progress.json"
OUT_RAW = DATA / "synth_code.jsonl"
OUT_SFT = DATA / "synth_code_sft.jsonl"
BAN_IMPORTS = ("os.system", "subprocess", "eval(", "exec(", "pickle",
               "socket", "requests", "urllib", "open(", "input()", "sys.stdin",
               "shutil", "ctypes", "multiprocessing", "threading")

TOPICS_ZH = [
    "数组遍历与统计", "字符串处理", "双指针", "哈希表计数", "简单数学计算",
    "递归基础", "排序与比较", "贪心策略", "链表基础操作", "位运算基础",
]
TOPICS_EN = [
    "array traversal & statistics", "string manipulation", "two pointers",
    "hashmap counting", "simple math", "basic recursion", "sorting & comparison",
    "greedy strategy", "linked list basics", "bitwise basics",
]

PROMPT_ZH = """你是算法讲师。请出一道中等偏简单的 Python 编程题，并完整示范解法。

严格按以下格式输出（不要输出其他内容）：

题目：<一句话描述题意 + 输入输出示例>
推理：<直接写这道题的解题思路：算法选择、复杂度、关键步骤，不要写你如何出题的过程>
代码：
```python
<完整可运行、无输入读取、纯函数形式的 Python 代码，含函数定义>
```
测试：
```python
<至少 3 条 assert 断言，覆盖示例和边界情况，无 print 干扰>
```

要求：代码不得使用 input()、文件、网络；测试只用 assert；题目主题是「{topic}」。"""

PROMPT_EN = """You are an algorithm instructor. Create an easy-to-medium Python
programming problem and demonstrate a complete solution.

Output strictly in this format (nothing else):

Question: <one-line description with input/output example>
Reasoning: <1-3 sentence step-by-step approach>
Code:
```python
<complete runnable, pure-function Python code, no input() calls>
```
Tests:
```python
<at least 3 assert statements covering the example and edge cases, no prints>
```

Requirements: no input(), no file/network IO; tests use assert only;
topic: {topic}"""

REJECT_SUBSTR = ("抱歉", "无法完成", "不能生成", "I cannot", "sorry",
                 "作为AI", "as an AI", "API", "语言模型", "model as")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOG / "synth_v2.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def call_ollama(prompt: str, num_predict: int = 1200) -> str:
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.6, "top_p": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))["response"]


def split_think(out: str) -> tuple[str, str]:
    """返回 (reasoning, body)。qwen3 推理内容在 <think>...</think>。"""
    reasoning = ""
    if "<think>" in out:
        m = re.search(r"<think>(.*?)</think>", out, re.S)
        if m:
            reasoning = m.group(1).strip()
            out = out.replace(m.group(0), "", 1)
    return reasoning, out.strip()


def extract_blocks(text: str) -> list[str]:
    """提取所有 ```python ... ``` 代码块。"""
    return [b.strip() for b in re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)]


def parse_output(out: str) -> dict | None:
    """解析教师输出为 {question, reasoning, code, tests}。
    reasoning 优先取正文「推理：」段（真解题思路）；think 元思考仅作兜底并去噪。"""
    _, body = split_think(out)
    blocks = extract_blocks(body)
    if len(blocks) < 2:
        return None
    code, tests = blocks[0], blocks[1]
    qm = re.search(r"题目[:：]\s*(.+)", body)
    question = qm.group(1).strip() if qm else ""
    # 兜底：题目 = 首个非空行
    if not question:
        for line in body.splitlines():
            if line.strip() and not line.strip().startswith(("代码", "测试", "```")):
                question = line.strip().strip("题目：:")
                break
    # 推理段：正文「推理：」或「Reasoning:」直到「代码：/Code:」止；无则用 think 去噪
    rm = re.search(r"(?:推理|Reasoning)[:：]\s*(.*?)(?=\n?\s*(?:代码|Code)[:：]|\Z)", body, re.S)
    reasoning = rm.group(1).strip() if rm else ""
    if not reasoning:
        think, _ = split_think(out)
        # think 去噪：丢弃「出题/我要出/确定题目/主题是」及英文出题计划的元思考行
        if think:
            meta = re.compile(
                r"出题|题目|主题|我要|现在要|首先|让我|"
                r"create.{0,20}problem|I need to|I want to|should be|let me|"
                r"make sure|start with|first,|topic|question should|"
                r"easy to medium|difficulty", re.I)
            lines = []
            for ln in think.splitlines():
                if meta.search(ln):
                    continue
                lines.append(ln)
            reasoning = "\n".join(lines).strip()
    if not (question and code and tests):
        return None
    return {"question": question, "reasoning": reasoning, "code": code, "tests": tests}


def verify(parsed: dict) -> bool:
    """本地沙箱执行 code+tests，rc==0 且无 Traceback 才通过。"""
    code, tests = parsed["code"], parsed["tests"]
    # 安全黑名单
    for bad in BAN_IMPORTS:
        if bad in code or bad in tests:
            return False
    if tests.count("assert") < 3:
        return False
    script = code + "\n\n" + tests
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=20,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    if r.returncode != 0 or "Traceback" in r.stderr:
        return False
    return True


def main() -> None:
    n_target = N_TARGET
    prog = {"done": 0, "pass": 0, "fail": 0, "req": 0, "start": time.time()}
    if PROG.exists():
        prog.update(json.loads(PROG.read_text(encoding="utf-8")))
        log(f"断点续跑：通过 {prog['pass']}/{n_target}，请求 {prog['req']}")

    raw = OUT_RAW.open("a", encoding="utf-8")
    sft = OUT_SFT.open("a", encoding="utf-8")

    # 题目去重：加载已有产出的题目进 seen（跨重启保持）
    seen_q = set()
    if OUT_RAW.exists():
        for ln in OUT_RAW.open(encoding="utf-8"):
            try:
                seen_q.add(json.loads(ln)["question"])
            except Exception:
                pass
    prog.setdefault("dup", 0)
    log(f"已见题目 {len(seen_q)} 个（去重用）")

    def save_prog():
        PROG.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")

    try:
        while prog["pass"] < n_target:
            # 主题轮换（zh 75%）
            use_zh = prog["req"] % 4 != 3
            topic = random.choice(TOPICS_ZH if use_zh else TOPICS_EN)
            prompt = (PROMPT_ZH if use_zh else PROMPT_EN).format(topic=topic)
            accepted = False
            for attempt in range(MAX_RETRY_PER_TOPIC + 1):
                prog["req"] += 1
                try:
                    out = call_ollama(prompt)
                except Exception as e:
                    log(f"请求失败: {e}，暂停 20s")
                    time.sleep(20)
                    continue
                parsed = parse_output(out)
                if parsed is None:
                    continue
                if any(w in (parsed["question"] + parsed["code"]) for w in REJECT_SUBSTR):
                    continue
                if verify(parsed):
                    accepted = True
                    break
            if accepted and parsed is not None:
                if parsed["question"] in seen_q:
                    prog["dup"] += 1
                    prog["fail"] += 1
                    continue
                seen_q.add(parsed["question"])
                sample = {
                    "id": f"synth_{prog['done']:05d}",
                    "src": "synth_v2", "lang": "zh" if use_zh else "en",
                    "topic": topic, "question": parsed["question"],
                    "reasoning": parsed["reasoning"], "code": parsed["code"],
                    "tests": parsed["tests"], "verified": True,
                }
                raw.write(json.dumps(sample, ensure_ascii=False) + "\n")
                asst = f"推理：{parsed['reasoning']}\n\n代码：\n```python\n{parsed['code']}\n```" \
                    if parsed["reasoning"] else f"代码：\n```python\n{parsed['code']}\n```"
                sft.write(json.dumps({"messages": [
                    {"role": "user", "content": parsed["question"]},
                    {"role": "assistant", "content": asst},
                ], "src": "synth_v2"}, ensure_ascii=False) + "\n")
                prog["done"] += 1
                prog["pass"] += 1
            else:
                prog["fail"] += 1

            if prog["req"] % 20 == 0:
                save_prog()
                el = (time.time() - prog["start"]) / 60
                log(f"req {prog['req']} | pass {prog['pass']}/{n_target} "
                    f"(通过率 {prog['pass']/max(1,prog['req'])*100:.0f}%) "
                    f"fail {prog['fail']} | {el:.0f} min")
    finally:
        save_prog()
        raw.close()
        sft.close()

    el = (time.time() - prog["start"]) / 60
    log(f"完成：通过 {prog['pass']} 条，请求 {prog['req']}，"
        f"通过率 {prog['pass']/max(1,prog['req'])*100:.0f}%，用时 {el:.0f} min")


if __name__ == "__main__":
    main()
