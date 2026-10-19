# -*- coding: utf-8 -*-
"""
SFT 前数据准备（一次性脚本）：
1. 从 data/synth_code.jsonl（原始全量）重新生成 data/synth_code_sft.jsonl，
   保证 600 条全量进训练（不依赖蒸馏进程的 sft 缓冲写入）。
2. 清洗英文样本 question 的 "Question: " 教师格式残留前缀。
3. 输出清洗统计，供核对。

用法：python prepare_sft.py [--min 600] [--raw data/synth_code_v2.jsonl] [--out data/synth_code_sft.jsonl]
"""
import json, re, sys
from pathlib import Path

ROOT = Path(r"F:\自研大模型")
RAW = ROOT / "data" / "synth_code.jsonl"
OUT = ROOT / "data" / "synth_code_sft.jsonl"

MIN = 600
if "--min" in sys.argv:
    MIN = int(sys.argv[sys.argv.index("--min") + 1])
if "--raw" in sys.argv:
    RAW = ROOT / sys.argv[sys.argv.index("--raw") + 1]
if "--out" in sys.argv:
    OUT = ROOT / sys.argv[sys.argv.index("--out") + 1]

Q_PREFIX = re.compile(r"^Question\s*[:：]\s*", re.I)

def main():
    if not RAW.exists():
        print(f"[错误] 找不到 {RAW}")
        sys.exit(1)

    samples = [json.loads(l) for l in RAW.open(encoding="utf-8") if l.strip()]
    print(f"raw 原始条数: {len(samples)}")
    if len(samples) < MIN:
        print(f"[警告] 原始条数 {len(samples)} < 目标 {MIN}，先不覆盖 sft 文件，蒸馏完成后再跑")
        sys.exit(0)

    cleaned = 0
    out_lines = []
    for d in samples:
        q = d["question"]
        if d.get("lang") == "en" and Q_PREFIX.match(q):
            q2 = Q_PREFIX.sub("", q).strip()
            if q2:
                q = q2
                cleaned += 1
        # 与 gen_synth_v2.py 的 sft 格式保持一致
        reasoning = d.get("reasoning") or ""
        asst = f"推理：{reasoning}\n\n代码：\n```python\n{d['code']}\n```" \
            if reasoning.strip() else f"代码：\n```python\n{d['code']}\n```"
        out_lines.append(json.dumps({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": asst},
            ],
            "src": "synth_v2",
        }, ensure_ascii=False))

    OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"重建完成: {OUT}")
    print(f"sft 条数: {len(out_lines)}")
    print(f"清洗英文 'Question:' 前缀: {cleaned} 条")
    # 核对：无重复 question、无空内容
    qs = [json.loads(l)["messages"][0]["content"] for l in out_lines]
    print(f"重复题目: {len(qs) - len(set(qs))}")
    print(f"空 user/assistant: {sum(1 for q in qs if not q.strip())} / "
          f"{sum(1 for l in out_lines if not json.loads(l)['messages'][1]['content'].strip())}")

if __name__ == "__main__":
    main()
