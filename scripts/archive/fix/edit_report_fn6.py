# -*- coding: utf-8 -*-
"""增量编辑：第六章补充函数级 pass@1 vs pass@8 证据表（表 6），
原表 6(复现命令链)→表 7、原表 7(GRPO)→表 8；更新 6.1 count_chars 与 6.3 正文。"""
import copy
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TARGET = r"F:\自研大模型\docs\自研106M大模型数据侧能力迭代研究报告.docx"
doc = Document(TARGET)


def find_para(text_start, style=None):
    for p in doc.paragraphs:
        if p.text.strip().startswith(text_start):
            if style is None or p.style.name == style:
                return p
    return None


# ---------- 1. 重编号：表 6→7（复现命令链）、表 7→8（GRPO），正文引用同步 ----------
p67 = find_para("表 6")
if p67:
    for r in p67.runs:
        r.text = r.text.replace("表 6", "表 7")
    print("表 6→表 7:", p67.text[:30])

p78 = find_para("表 7")
if p78:
    for r in p78.runs:
        r.text = r.text.replace("表 7", "表 8")
    print("表 7→表 8:", p78.text[:30])

# 正文"如表 7 所示"→"如表 8 所示"（GRPO 引言）
for p in doc.paragraphs:
    if "结果矩阵如表 7 所示" in p.text:
        for r in p.runs:
            r.text = r.text.replace("如表 7 所示", "如表 8 所示")
        print("正文引用更新: 如表 8 所示")
        break

# ---------- 2. 更新 6.1 count_chars 诊断（死题→部分命中） ----------
p61 = find_para("评测中 count_chars 4 题")
if p61:
    for r in p61.runs:
        r.text = ""
    p61.runs[0].text = (
        "评测中 count_chars 4 题与 count_characters 8 题语义完全同质（均为“统计字符次数 + 返回字典”），"
        "题面措辞无任何可区分信号，且训练数据中 count_characters 变体占比显著更高，"
        "模型只能学好一个名字（保大弃小）。采样维度复核显示 count_chars 4 题 pass@8 命中 1/4，"
        "并非完全死题——模型对同名函数已有部分能力，瓶颈在名字竞争与样本占比。")
    print("6.1 更新完成")

# ---------- 3. 更新 6.3 正文（精确 13 函数数据） ----------
p63 = find_para("值得注意的是，上述顽固函数")
if p63:
    for r in p63.runs:
        r.text = ""
    p63.runs[0].text = (
        "按函数名聚合 100 题（51 个函数）后，顽固区的采样命中全景比定性描述更清晰："
        "13 个函数 pass@1 全挂但 pass@8 有命中（见表 6），其中 process_string 4/4、fib 4/4、"
        "sum_array 2/2、convert_to_ascii 2/2、count_set_bits 2/2 全部多采样命中——"
        "说明这些函数的能力已通过数据工程灌入权重，pass@1 仅受单采样确定性限制，"
        "模型真实能力被 pass@1 低估。其余双零死题（30 个函数）中，merge 系 5 题与大量单题稀有函数"
        "（sum_list/sum_even/flip_bits 等）受样本不足与函数名竞争共同制约，是评测集版本迭代的主要对象。")
    print("6.3 更新完成")

# ---------- 4. 插入表 6：多采样命中函数清单（52 行后、表 5 前） ----------
anchor = find_para("表 5")
if anchor is None:
    raise SystemExit("未找到表 5 caption")

ms_rows = [
    ("count_chars", 4, 1), ("process_string", 4, 4), ("fib", 4, 4),
    ("reverse_linked_list", 3, 1), ("count_words", 3, 1), ("sum_array", 2, 2),
    ("sum_even_numbers", 2, 1), ("convert_to_ascii", 2, 2), ("count_set_bits", 2, 2),
    ("multiply", 1, 1), ("count_even_numbers", 1, 1), ("count_even", 1, 1),
    ("flip_bit", 1, 1),
]

# caption 段落（复制 anchor 样式=Caption/Normal）
cap = copy.deepcopy(anchor._p)
for child in list(cap):
    if child.tag == qn("w:r"):
        cap.remove(child)
rpr = None
for child in anchor._p:
    if child.tag == qn("w:r"):
        r0 = child.find(qn("w:rPr"))
        if r0 is not None:
            rpr = copy.deepcopy(r0)
        break
r = OxmlElement("w:r")
if rpr is not None:
    r.append(rpr)
t = OxmlElement("w:t")
t.text = "表 6  多采样命中函数清单（pass@1 = 0 但 pass@8 > 0，v8）"
t.set(qn("xml:space"), "preserve")
r.append(t)
cap.append(r)
anchor._p.addprevious(cap)

# 表格
tb = doc.add_table(rows=len(ms_rows) + 1, cols=3)
tb.style = "Table Grid"
tb.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, h in enumerate(("函数", "评测题数", "pass@8 命中")):
    c = tb.rows[0].cells[j]
    c.text = ""
    p = c.paragraphs[0]
    rr = p.add_run(h)
    rr.bold = True
    rr.font.size = Pt(10.5)
    rr.font.name = "Times New Roman"
    rr._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    tcPr = c._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "D9D9D9")
    tcPr.append(shd)
for i, (fn, n, hit) in enumerate(ms_rows):
    for j, val in enumerate((fn, str(n), str(hit))):
        c = tb.rows[i + 1].cells[j]
        c.text = ""
        p = c.paragraphs[0]
        rr = p.add_run(val)
        rr.font.size = Pt(10.5)
        rr.font.name = "Times New Roman"
        rr._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
for row in tb.rows:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
tb.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
for j, w in enumerate((Cm(4.5), Cm(3.0), Cm(3.5))):
    for cell in tb.columns[j].cells:
        cell.width = w
# 移动表格到 caption 后
anchor._p.addprevious(tb._tbl)
print("表 6 插入完成（13 函数）")

doc.save(TARGET)
print("保存完成")
