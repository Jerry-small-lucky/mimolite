# -*- coding: utf-8 -*-
"""增量编辑研究报告：插入"八、GRPO 训练侧实验（负结果）"章节 + 结论微调 + 档案清单补充"""
import copy
import docx
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TARGET = r"F:\自研大模型\docs\自研106M大模型数据侧能力迭代研究报告.docx"
doc = Document(TARGET)


def find_heading(text):
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading") and p.text.strip() == text:
            return p
    return None


def clone_paragraph(ref_p, text, style=None):
    """深拷贝段落 XML（继承样式与 rPr），替换文本为新 run。"""
    new_el = copy.deepcopy(ref_p._p)
    # 清空所有 run，保留 pPr
    for child in list(new_el):
        if child.tag == qn("w:r"):
            new_el.remove(child)
    # 以第一个 run 的 rPr 为基准
    rpr = None
    for child in ref_p._p:
        if child.tag == qn("w:r"):
            r0 = child.find(qn("w:rPr"))
            if r0 is not None:
                rpr = copy.deepcopy(r0)
            break
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    new_el.append(r)
    return new_el


def insert_before(anchor_p, el):
    anchor_p._p.addprevious(el)


def add_body_before(anchor_p, text):
    ref = anchor_p  # 用 Heading 段落作为格式基准，再改成 Normal 文本样式
    el = clone_paragraph(ref, text)
    # 去掉标题样式中的编号/加粗继承风险：改用 Normal
    pPr = el.find(qn("w:pPr"))
    if pPr is not None:
        pStyle = pPr.find(qn("w:pStyle"))
        if pStyle is not None:
            pStyle.set(qn("w:val"), "a3")  # 内置 Normal styleId 通常为 a3 或 Normal
    insert_before(anchor_p, el)


# ---------- 1. 定位锚点 ----------
c8 = find_heading("八、结论与下一步")
if c8 is None:
    raise SystemExit("未找到『八、结论与下一步』标题")
print("锚点定位:", c8.style.name, "|", c8.text)

# ---------- 2. 插入新章节（在 c8 前） ----------
h1 = clone_paragraph(c8, "八、GRPO 训练侧实验（负结果）")
insert_before(c8, h1)

# 8.1
for h2_text in ("8.1 动机与设计", "8.2 四轮实验矩阵", "8.3 归因与结论"):
    h2 = clone_paragraph(c8, h2_text)
    # 改样式为 Heading 2
    pPr = h2.find(qn("w:pPr"))
    pStyle = pPr.find(qn("w:pStyle"))
    if pStyle is not None:
        pStyle.set(qn("w:val"), "a4")  # Heading 2 styleId
    insert_before(c8, h2)

b1 = ("为突破 SFT 数据上限、攻坚 pass@1 与顽固区，实现了 grpo_code.py：以代码沙箱可验证奖励为信号"
      "（断言通过 +1.0、无代码块 -0.5），policy + 冻结参考模型做 KL 锚定，组内相对优势更新；"
      "从当前最佳 sft_v8 权重初始化。过程中修复了早期 grpo.py 骨架的标量梯度缺陷（原版无法反向传播）。")
insert_before(c8, clone_paragraph(c8, b1))

b2 = ("四轮实验均在评测集 32 题上以低温采样（temp 0.2）评估，RL 前基线 pass@1 = 56.2%。"
      "结果矩阵如表 7 所示：模板题对 v8 过拟合导致无学习信号；筛选题信号不足；最难题上模型 rollout 全错，"
      "GRPO 退化为惩罚一切，评测跌至 43.8%。")
insert_before(c8, clone_paragraph(c8, b2))

# ---------- 3. 表格 7（GRPO 实验矩阵） ----------
t = doc.add_table(rows=5, cols=5)
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["轮次", "训练数据", "关键参数", "终局评测", "结论"]
rows = [
    ["v1", "模板题（v8 全会）", "kl 0.01 / lr 1e-5 / temp 1.0", "50.0%", "策略漂移，破坏基线"],
    ["v2", "模板题", "kl 0.1 / lr 3e-6 / temp 0.7", "56.2%", "无信号（rollout 全过）"],
    ["v3", "筛选题 264（通过率≤0.5）", "同上", "56.2%", "信号不足（96 rollout）"],
    ["v4", "最难题 215（通过率=0）", "同上 + 60 组", "43.8%", "惩罚一切，破坏能力"],
]
for j, h in enumerate(headers):
    c = t.rows[0].cells[j]
    c.text = ""
    p = c.paragraphs[0]
    r = p.add_run(h)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    tcPr = c._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "D9D9D9")
    v = tcPr.find(qn("w:vAlign"))
    if v is not None:
        v.addprevious(shd)
    else:
        tcPr.append(shd)
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = t.rows[i + 1].cells[j]
        c.text = ""
        p = c.paragraphs[0]
        r = p.add_run(val)
        r.font.size = Pt(10.5)
        r.font.name = "Times New Roman"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in (0, 3) else WD_ALIGN_PARAGRAPH.LEFT
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
for r in t.rows:
    trPr = r._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
# 表题（插在表格前）
cap = clone_paragraph(c8, "表 7  GRPO 四轮实验矩阵（评测 32 题 pass@1，RL 前 56.2%）")
insert_before(c8, cap)
# 移动表格 XML 到 c8 之前（表格默认加在文档末尾）
c8._p.addprevious(t._tbl)

# 8.3 正文
b3 = ("归因有三：其一，能力上限——106M 在最难题上连接近正确的 rollout 都产生不了，RL 缺少可强化的正样本；"
      "其二，连坐效应——惩罚失败代码的 token 会连带削弱与成功代码共享的语法与函数名模式；"
      "其三，规模差距——MiMo-7B 等成功案例依赖 13 万道可验证题的采样池与更大模型容量，"
      "8GB 显存限制了 group-size 与训练组数。")
insert_before(c8, clone_paragraph(c8, b3))

b4 = ("结论：在 106M + 8GB 规模下小规模 GRPO 暂不可行，先搁置。资产全部保留（grpo_code.py、"
      "OOD 题筛选脚本、data/grpo_train.jsonl 与 grpo_hard.jsonl、runs/grpo_code_v1~v4 四轮档案），"
      "模型规模升级后可复用。")
insert_before(c8, clone_paragraph(c8, b4))

# ---------- 4. 原第八章改编号 + 结论正文微调 ----------
for r in c8.runs:
    if "八、结论" in r.text:
        r.text = r.text.replace("八、结论", "九、结论")
    elif r.text.strip() == "八、结论与下一步":
        r.text = "九、结论与下一步"
if c8.text == "八、结论与下一步":
    c8.runs[0].text = "九、结论与下一步"

# 结论正文第二段：GRPO 表述更新
new_text = ("下一步方向：其一，数据侧残余增量与评测集版本迭代（顽固区已归因于评测设计缺陷，"
            "可推动评测集完善）；其二，将本报告的实验矩阵与负结果（含第八章 GRPO 验证）作为研究叙事交付，"
            "形成敲门砖材料。GRPO 训练侧已实证在 106M 规模暂不可行，待模型规模升级后重启。")
for p in doc.paragraphs:
    if p.style.name == "Normal" and "下一步两个方向" in p.text:
        for r in p.runs:
            r.text = ""
        p.runs[0].text = new_text
        print("结论正文已更新")
        break

# ---------- 5. 7.2 档案清单补 GRPO 资产 ----------
for p in doc.paragraphs:
    if p.style.name == "Normal" and "分析脚本：compare_v5_v6.py" in p.text:
        extra = "。GRPO 训练侧资产：grpo_code.py（代码沙箱奖励版）、select_grpo_data.py/filter_hard.py（OOD 题筛选）、data/grpo_train.jsonl 与 grpo_hard.jsonl、runs/grpo_code_v1~v4（四轮实验档案）"
        last = p.runs[-1]
        last.text = last.text + extra
        print("7.2 档案清单已补充")
        break

doc.save(TARGET)
print("保存完成:", TARGET)
