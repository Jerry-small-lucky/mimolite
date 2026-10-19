# -*- coding: utf-8 -*-
"""README.md 更新：加入 P0 标准基准对照 + P1 多 seed 稳健性 + GGUF demo + 研究提案链接。
在『〇、当前主线与核心成果』区块追加内容。用法：python scripts/update_readme.py
"""
import json
from pathlib import Path

ROOT = Path(r"F:\自研大模型")
README = ROOT / "README.md"
LOGS = ROOT / "logs"


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(d):
    m = d["meta"]
    return f"{m['n_pass']}/{m['n']} ({m['pass_rate']*100:.0f}%)"


def pctx(d, fallback):
    if d is None or not d["meta"].get("done"):
        return fallback
    return pct(d)


he_s1, mb_s1 = load(LOGS / "std_humaneval_s1.json"), load(LOGS / "std_mbpp_s1.json")
he_s8 = load(LOGS / "std_humaneval_s8.json")
_mb8 = LOGS / "std_mbpp_s8.json"
mb_s8 = load(_mb8) if _mb8.exists() else None
mb8s = pctx(mb_s8, "待补（后台运行中）")

text = README.read_text(encoding="utf-8")

anchor = "- **研究文档**：`docs/自研106M大模型数据侧能力迭代研究报告.docx`"
assert anchor in text

block = f"""- **外部基准锚点（P0，2026-10）**：在标准英文基准上做外分布对照，验证泛化边界。HumanEval pass@1 = **{pct(he_s1)}**、pass@8 = **{pct(he_s8)}**；MBPP pass@1 = **{pct(mb_s1)}**、pass@8 = **{mb8s}**。与内分布（自建 100 题 29/57）构成**泛化鸿沟**：数据侧迭代把能力绑定在训练分布（中文指令 × 自建题面 × 自建函数名），尚未形成跨分布泛化。判定经逐例诊断确认真实，评测口径（指令式适配、低温单采样）已在研究报告中披露。
- **多 seed 稳健性（P1）**：同一数据链配置以 3 个随机种子重复（seed 7/8/9），自建 pass@1 = **29 / 31 / 24（均值 28%）**，确认非单点噪声；完整对照（含 pass@8）见研究报告 4.5。
- **研究提案**：`docs/研究提案-预算守恒与函数名竞争.docx`（预算守恒律 / 函数名竞争律 / 多采样恢复，2 页可评审提案）。
- **GGUF demo**：`out/mimolite-sftv8.gguf`（221 MB，llama.cpp 已验证可加载推理，约 724 tok/s）。
"""
text = text.replace(anchor, anchor + "\n" + block)
README.write_text(text, encoding="utf-8")
print("README 已更新")
