# -*- coding: utf-8 -*-
"""数据侧迭代 A：构建无泄漏训练集（支持多 seed 模板）
剔除评测集 100 题（eval_src_lines）后，合并蒸馏 510 条 + 模板 N 条 -> synth_code_v2[_seedN].jsonl
用法：
  python scripts/build_train_v2.py                        # 默认：data/synth_code_tpl.jsonl -> synth_code_v2.jsonl
  python scripts/build_train_v2.py --tpl data/synth_code_tpl_seed8.jsonl --out data/synth_code_v2_seed8.jsonl
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(r"F:\自研大模型")
Q_PREFIX = re.compile(r"^Question\s*[:：]\s*", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpl", type=str, default=str(ROOT / "data/synth_code_tpl.jsonl"))
    ap.add_argument("--out", type=str, default=str(ROOT / "data/synth_code_v2.jsonl"))
    args = ap.parse_args()
    tpl_p, out_p = Path(args.tpl), Path(args.out)

    meta = json.load(open(ROOT / "logs/eval_set_meta.json", encoding="utf-8"))
    eval_lines = set(meta["eval_src_lines"])
    print(f"评测集行索引（剔除）: {len(eval_lines)} 条")

    old = [json.loads(l) for l in open(ROOT / "data/synth_code.jsonl", encoding="utf-8")]
    tpl = [json.loads(l) for l in open(tpl_p, encoding="utf-8")]

    train_old = [(i, d) for i, d in enumerate(old) if i not in eval_lines]
    print(f"蒸馏保留: {len(train_old)}（原 {len(old)}，剔除 {len(old) - len(train_old)} 评测题）")

    def clean_q(d):
        q = d["question"]
        if d.get("lang") == "en" and Q_PREFIX.match(q):
            q = Q_PREFIX.sub("", q).strip()
        return q

    eval_qs = {clean_q(json.loads(l)) for l in open(ROOT / "data/eval_code.jsonl", encoding="utf-8")}
    leak = [d["question"] for _, d in train_old if clean_q(d) in eval_qs]
    tpl_leak = [d["question"] for d in tpl if clean_q(d) in eval_qs]
    print(f"泄漏校验（蒸馏部分）: {len(leak)} | （模板部分）: {len(tpl_leak)}")

    merged = [d for _, d in train_old] + tpl
    with open(out_p, "w", encoding="utf-8") as f:
        for d in merged:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"训练集构建完成: {len(merged)} 条（蒸馏 {len(train_old)} + 模板 {len(tpl)}）-> {out_p}")


if __name__ == "__main__":
    main()
