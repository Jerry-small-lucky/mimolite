# -*- coding: utf-8 -*-
"""修复 tbl[4]（表 6）表头 tcPr 子节点顺序：tcW < shd < vAlign"""
from docx import Document
from docx.oxml.ns import qn

TARGET = r"F:\自研大模型\docs\自研106M大模型数据侧能力迭代研究报告.docx"
doc = Document(TARGET)
tb = doc.tables[4]  # audit table_index 4 = 多采样命中表

def order_tcpr(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    nodes = list(tcPr)
    # 重排为 tcW, gridSpan, hMerge, vMerge, shd, vAlign, ...
    order = {qn("w:tcW"): 0, qn("w:gridSpan"): 1, qn("w:hMerge"): 2, qn("w:vMerge"): 3,
             qn("w:shd"): 4, qn("w:vAlign"): 5}
    nodes.sort(key=lambda n: order.get(n.tag, 9))
    for n in nodes:
        tcPr.remove(n)
    for n in nodes:
        tcPr.append(n)

for cell in tb.rows[0].cells:
    order_tcpr(cell)

doc.save(TARGET)
print("表 6 表头 tcPr 顺序修复完成")
