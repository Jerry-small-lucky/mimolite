"""
GRPO 奖励函数单元测试
=====================

可验证奖励是 RL 训练的地基：抽取错一个字符，奖励信号就全错。
这里覆盖 GSM8K 常见答案格式（纯数字、千分位、货币、小数、无答案）。

用法： python test_grpo_reward.py
"""

from __future__ import annotations

import sys

import grpo

CASES = [
    # (completion, gold, expected_reward, 说明)
    ("Let me think... 12/2=6. Then 24+6=30.\n<answer>30</answer>", "30", 1.2, "标准 CoT + 答案"),
    ("So the total is 30.\n<answer>30</answer>", "30", 1.2, "简短推导"),
    ("Thus 42 items.\n<answer>42</answer>", "30", 0.2, "答案错误但格式规范"),
    ("I think maybe 30 or 31.", "30", 0.0, "无标签，靠兜底抽取"),
    ("no answer here at all", "30", -0.5, "完全没给答案"),
    ("work...\n<answer>1,024</answer>", "1024", 1.2, "千分位"),
    ("x = 2.50\n<answer>2.5</answer>", "2.5", 1.2, "小数"),
    ("reasoning\n<answer>72</answer>", "72", 1.2, "纯整数"),
]


def main() -> int:
    failures = 0
    print("=" * 74)
    print("GRPO 可验证奖励单元测试")
    print("=" * 74)
    print(f"{'':4}{'说明':<20}{'得到':>7}{'期望':>7}{'':4} 明细")
    for comp, gold, expect, desc in CASES:
        r, d = grpo.compute_reward(comp, gold)
        good = abs(r - expect) < 1e-6
        failures += 0 if good else 1
        print(f"{'OK' if good else 'BAD':<4}{desc:<20}{r:>7.2f}{expect:>7.2f}   "
              f"  {d}")

    print("\n--- 答案抽取 ---")
    checks = [
        ("<answer>  77  </answer> 尾部", grpo.extract_answer("foo\n<answer>  77  </answer> tail"), "77"),
        ("无标签时兜底取末行", grpo.extract_answer("reasoning...\nthe answer is 88."), "88"),
        ("完全无答案", grpo.extract_answer("blah blah"), None),
    ]
    for desc, got, want in checks:
        good = got == want
        failures += 0 if good else 1
        print(f"{'OK' if good else 'BAD':<4}{desc:<20} 得到={got!r} 期望={want!r}")

    print("\n--- 数值归一化 ---")
    for raw, want in [("1,024", "1024"), ("$30", "30"), ("30.0", "30"),
                      ("2.50", "2.5"), ("  18  ", "18")]:
        got = grpo.norm_num(raw)
        good = got == want
        failures += 0 if good else 1
        print(f"{'OK' if good else 'BAD':<4}{raw!r:<12} -> {got!r}  期望 {want!r}")

    print("\n" + "=" * 74)
    if failures:
        print(f"失败 {failures} 项")
        return 1
    print("全部用例通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())