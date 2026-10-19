# -*- coding: utf-8 -*-
"""研究报告增量编辑：新增 1.4 与既有工作的定位（Related Work 简版）。
流程：备份为 SOURCE -> 在 1.3 正文后插入 1.4 标题+正文 -> 保存 TARGET（原文件）。
"""
import copy
import shutil
import time
from pathlib import Path
import docx

SRC = Path(r"F:\自研大模型\docs\自研106M大模型数据侧能力迭代研究报告.docx")
TS = time.strftime("%Y%m%d_%H%M%S")
BACKUP = Path(rf"F:\自研大模型\docs\自研106M大模型数据侧能力迭代研究报告_backup_{TS}.docx")
shutil.copy2(SRC, BACKUP)

doc = docx.Document(str(SRC))

# 定位：1.3 标题(13) 正文(14) -> 在其后插入 1.4
anchor = doc.paragraphs[14]  # 1.3 正文段

h2 = docx.oxml.OxmlElement("w:p")
# 复制 1.3 标题段格式
t13 = doc.paragraphs[13]._p
new_h = copy.deepcopy(t13)
# 清空 runs 文本
for r in new_h.findall(docx.oxml.ns.qn("w:r")):
    new_h.remove(r)
# 写入新标题文本（保留样式）
r = docx.oxml.OxmlElement("w:r")
t = docx.oxml.OxmlElement("w:t")
t.text = "1.4 与既有工作的定位"
r.append(t)
new_h.append(r)

# 正文段 1：复制 1.3 正文段格式
new_b1 = copy.deepcopy(doc.paragraphs[14]._p)
for r in new_b1.findall(docx.oxml.ns.qn("w:r")):
    new_b1.remove(r)
r1 = docx.oxml.OxmlElement("w:r")
t1 = docx.oxml.OxmlElement("w:t")
t1.text = ("主流代码模型的数据策略可分为四类。其一，大规模语料混合：DeepSeek-Coder 以 87% 源码、"
           "10% 英文代码语料与 3% 中文语料预训练，并用 GRPO 做代码偏好对齐（Guo et al., arXiv:2401.14196）。"
           "其二，教师合成：phi-1 用 GPT-4 生成的教科书质量数据训练 1.3B 模型（Gunasekar et al., arXiv:2306.11644）。"
           "其三，自举：STaR 让模型用自身正确推理微调（Zelikman et al., arXiv:2203.14465）。"
           "其四，自生成指令：Self-Instruct 用模型自产指令扩展微调集（Wang et al., arXiv:2212.10560）。")
r1.append(t1)
new_b1.append(r1)

# 正文段 2
new_b2 = copy.deepcopy(doc.paragraphs[14]._p)
for r in new_b2.findall(docx.oxml.ns.qn("w:r")):
    new_b2.remove(r)
r2 = docx.oxml.OxmlElement("w:r")
t2 = docx.oxml.OxmlElement("w:t")
t2.text = ("上述路线的共同隐含假设是存在强教师、或数据规模与算力足够。本研究占据的约束组合——106M 参数、"
           "禁止教师蒸馏、600 条规则化合成预算、8GB 单卡——与 MiMo-V2.5 等百亿级模型后训练（48T token）"
           "相比规模相差 5 个数量级，但固定预算下的样本-函数分配问题在小模型上反而更尖锐："
           "预算守恒律与函数名竞争律直接决定能力边界（见第五、六章）。")
r2.append(t2)
new_b2.append(r2)

anchor._p.addnext(new_b2)
new_b2.addnext(new_h)  # 顺序：h 在 b2 前 -> b2 在 h 后；需 h->b1->b2
# 上面顺序错了，重排：anchor -> h -> b1 -> b2
# 先移除再按序插入
anchor._p.addnext(new_h)
new_h.addnext(new_b1)
new_b1.addnext(new_b2)

doc.save(str(SRC))
print("已插入 1.4 章节 ->", SRC)
print("备份 ->", BACKUP)
