# -*- coding: utf-8 -*-
"""修复新增章节的样式与标点"""
from docx import Document
from docx.shared import Pt

TARGET = r"F:\自研大模型\docs\自研106M大模型数据侧能力迭代研究报告.docx"
doc = Document(TARGET)

FIX_H2 = {"8.1 动机与设计", "8.2 四轮实验矩阵", "8.3 归因与结论"}
FIX_NORMAL_STARTS = ("为突破 SFT 数据上限", "四轮实验均在评测集", "归因有三", "结论：在 106M")

for p in doc.paragraphs:
    t = p.text.strip()
    if t in FIX_H2 and p.style.name != "Heading 2":
        p.style = doc.styles["Heading 2"]
        print("H2:", t)
    elif t.startswith(FIX_NORMAL_STARTS) and p.style.name != "Normal":
        p.style = doc.styles["Normal"]
        p.paragraph_format.first_line_indent = Pt(24)
        p.paragraph_format.line_spacing_rule = 3  # 1.5 倍
        print("Normal:", t[:20])

# 修复重复句号
for p in doc.paragraphs:
    if "。。" in p.text:
        for r in p.runs:
            if "。。" in r.text:
                r.text = r.text.replace("。。", "。")
        print("标点修复:", p.text[:40])

doc.save(TARGET)
print("样式修复完成")
