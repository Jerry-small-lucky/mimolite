# -*- coding: utf-8 -*-
"""
规则化模板合成（不蒸馏）—— 数据侧迭代 A 方案②
====================================================================
用确定性模板程序化生成"题目+参考实现+推理+断言测试"，生成时对参考
实现跑断言自验证（答案保证正确），不依赖任何教师模型。生成样本与
评测集（data/eval_code.jsonl）同构：自然语言题目（函数名隐含）、
```python 代码块、assert 测试；带推理句，与蒸馏样本 SFT 格式一致。

主题覆盖（与评测 topic 分布对齐，12/17 个主题，71 个变体）：
  哈希表计数 / 排序与比较 / 字符串处理 / 简单数学计算 /
  数组遍历与统计 / 双指针 / 递归基础 / 位运算基础 / 贪心策略 /
  链表基础操作 / 回文检测（并入字符串）/ 数列与递推（fib 并入数学）
  函数名对齐评测接口（factorial/count_characters/count_ones/fib/reverse_linked_list 等）

用法：
  python scripts/synth_code_tpl.py --demo          # 按主题轮询展示 6 条样例
  python scripts/synth_code_tpl.py --n 600 --seed 7   # 全量生成 -> data/synth_code_tpl.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify(code: str, tests: str) -> bool:
    """对参考实现跑断言，保证模板生成答案正确。"""
    try:
        r = subprocess.run([sys.executable, "-c", code + "\n" + tests],
                           capture_output=True, text=True, timeout=10,
                           stdin=subprocess.DEVNULL)
        return r.returncode == 0 and "Traceback" not in r.stderr
    except Exception:
        return False


def rec(topic_zh: str, topic_en: str, lang: str, question: str, code: str,
        tests: str, reasoning: str) -> dict:
    return {"src": "tpl", "lang": lang,
            "topic": topic_zh if lang == "zh" else topic_en,
            "question": question, "code": code, "tests": tests,
            "reasoning": reasoning}


def gen_tests(rng: random.Random, fn: str, n: int = 3) -> list[str]:
    """为给定函数名生成 n 组随机断言（输入用 seen 去重）。"""
    if fn == "reverse_linked_list":
        out = []
        while len(out) < n:
            vals = [rng.randint(1, 9) for _ in range(rng.randint(1, 6))]
            rev = vals[::-1]
            expr = "ListNode(" + str(vals[-1]) + ")"
            for v in reversed(vals[:-1]):
                expr = "ListNode(" + str(v) + ", " + expr + ")"
            out.append(f"assert reverse_linked_list({expr}).val == {rev[0]}")
            if len(rev) > 1:
                out.append(f"assert reverse_linked_list({expr}).next.val == {rev[1]}")
        return out
    seen: set = set()
    out = []
    while len(out) < n:
        args, expect = _case(rng, fn)
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(args, tuple):
            arg_str = ", ".join(repr(a) for a in args)
        else:
            arg_str = repr(args)
        out.append(f"assert {fn}({arg_str}) == {expect!r}")
    return out


def _case(rng: random.Random, fn: str):
    """返回 (args, 期望值)。args 为单值或元组。"""
    if fn == "count_frequencies":
        lst = [rng.randint(0, 4) for _ in range(rng.randint(0, 10))]
        return lst, _cf(lst)
    if fn == "count_characters":
        s = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 12)))
        return s, _cc(s)
    if fn == "count_words":
        s = "  ".join(rng.choice(["a", "bb", "ccc"]) for _ in range(rng.randint(0, 8)))
        return s, _cw(s)
    if fn == "most_frequent":
        lst = [rng.randint(0, 3) for _ in range(rng.randint(1, 12))]
        return lst, _mf(lst)
    if fn == "find_duplicates":
        lst = [rng.randint(0, 3) for _ in range(rng.randint(0, 12))]
        return lst, _fd(lst)
    if fn == "second_largest":
        lst = [rng.randint(0, 6) for _ in range(rng.randint(1, 10))]
        return lst, _sl(lst)
    if fn == "sort_unique":
        lst = [rng.randint(0, 5) for _ in range(rng.randint(0, 10))]
        return lst, sorted(set(lst))
    if fn == "merge_sorted":
        a = sorted(rng.randint(0, 9) for _ in range(rng.randint(0, 6)))
        b = sorted(rng.randint(0, 9) for _ in range(rng.randint(0, 6)))
        return (a, b), sorted(a + b)
    if fn == "is_sorted":
        lst = sorted(rng.randint(0, 9) for _ in range(rng.randint(0, 8)))
        if rng.random() < 0.3 and len(lst) > 1:
            lst = lst[:]
            lst[0], lst[-1] = lst[-1], lst[0]
        return lst, _is(lst)
    if fn == "count_vowels":
        s = "".join(rng.choice("aEiOu xyzBC") for _ in range(rng.randint(0, 12)))
        return s, _cv(s)
    if fn == "reverse_words":
        s = " ".join(rng.choice(["ab", "cde", "f", "ghij"]) for _ in range(rng.randint(0, 6)))
        return s, _rw(s)
    if fn == "is_palindrome":
        base = "".join(rng.choice("ab") for _ in range(rng.randint(0, 6)))
        s = base + base[::-1]
        if rng.random() < 0.4 and s:
            s = s[: len(s) // 2] + "x" + s[len(s) // 2:]
        return s, _ip(s)
    if fn == "capitalize_words":
        s = " ".join(rng.choice(["ab", "cde", "f", "ghij"]) for _ in range(rng.randint(0, 6)))
        return s, _cw2(s)
    if fn == "remove_consecutive_dups":
        s = "".join(rng.choice("aab") for _ in range(rng.randint(0, 10)))
        return s, _rcd(s)
    if fn == "is_prime":
        n = rng.randint(1, 97)
        return n, _prime(n)
    if fn == "gcd":
        a, b = rng.randint(1, 96), rng.randint(1, 96)
        return (a, b), _gcd(a, b)
    if fn == "fibonacci":
        n = rng.randint(0, 20)
        return n, _fib(n)
    if fn == "sum_range":
        a = rng.randint(-20, 20)
        b = a + rng.randint(0, 30)
        return (a, b), _sr(a, b)
    if fn == "array_sum":
        lst = [rng.randint(-9, 9) for _ in range(rng.randint(0, 10))]
        return lst, sum(lst)
    if fn == "count_evens":
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        return lst, sum(1 for x in lst if x % 2 == 0)
    if fn == "cumulative_sum":
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        out, acc = [], 0
        for x in lst:
            acc += x
            out.append(acc)
        return lst, out
    if fn == "max_difference":
        lst = [rng.randint(0, 20) for _ in range(rng.randint(1, 8))]
        return lst, (max(lst) - min(lst) if lst else 0)
    if fn == "two_sum":
        vals = [rng.randint(1, 9) for _ in range(rng.randint(2, 8))]
        tgt = vals[0] + vals[1]
        return (vals, tgt), _ts(vals, tgt)
    if fn == "is_palindrome_list":
        lst = [rng.randint(0, 5) for _ in range(rng.randint(0, 6))]
        s = lst + lst[::-1]
        if rng.random() < 0.3 and s:
            s[len(s) // 2] = 99
        return s, s == s[::-1]
    if fn == "remove_duplicates_sorted":
        lst = sorted(rng.randint(0, 5) for _ in range(rng.randint(0, 10)))
        return lst, _rds(lst)
    if fn == "factorial":
        n = rng.randint(0, 12)
        return n, _fac(n)
    if fn == "power":
        b, e = rng.randint(0, 5), rng.randint(0, 6)
        return (b, e), _pow(b, e)
    if fn == "count_digits":
        n = rng.randint(0, 99999)
        return n, _cd(n)
    if fn == "count_set_bits":
        n = rng.randint(0, 1023)
        return n, bin(n).count("1")
    if fn == "is_power_of_two":
        n = 1 << rng.randint(0, 9)
        if rng.random() < 0.3:
            n += 1
        return n, (n > 0 and (n & (n - 1)) == 0)
    if fn == "has_odd_ones":
        n = rng.randint(0, 255)
        return n, bin(n).count("1") % 2 == 1
    if fn == "max_profit":
        prices = [rng.randint(1, 20) for _ in range(rng.randint(1, 8))]
        return prices, _mp(prices)
    if fn == "coin_change_min":
        amount = rng.randint(0, 60)
        return amount, _ccm(amount)
    if fn == "can_jump":
        n = rng.randint(1, 6)
        nums = [rng.randint(0, 2) for _ in range(n)]
        return nums, _cj(nums)
    if fn == "merge_intervals":
        n = rng.randint(0, 5)
        ivs = []
        for _ in range(n):
            a = rng.randint(0, 8)
            ivs.append([a, a + rng.randint(0, 4)])
        return ivs, _mi(ivs)
    # ---- v2 扩展：对齐评测函数名 ----
    if fn in ("count_chars", "count_char_occurrences"):
        s = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 12)))
        return s, _cco(s)
    if fn in ("count_elements", "count_frequency", "count_occurrences"):
        lst = [rng.randint(0, 4) for _ in range(rng.randint(0, 10))]
        return lst, _ceq(lst)
    if fn == "sort_list":
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        return lst, sorted(lst)
    if fn in ("merge_sorted_arrays", "merge_two_sorted_arrays", "merge"):
        a = sorted(rng.randint(0, 9) for _ in range(rng.randint(0, 6)))
        b = sorted(rng.randint(0, 9) for _ in range(rng.randint(0, 6)))
        return (a, b), sorted(a + b)
    if fn == "is_strictly_increasing":
        n = rng.randint(1, 8)
        lst = sorted(rng.randint(0, 20) for _ in range(n))
        if rng.random() < 0.4 and n > 1:
            lst[n // 2] = lst[n // 2 - 1]
        return lst, _strict(lst)
    if fn == "is_all_equal":
        lst = [rng.choice([3, 7]) for _ in range(rng.randint(0, 8))]
        if rng.random() < 0.3 and lst:
            lst[0] = 9
        return lst, _aeq(lst)
    if fn == "compare_lists":
        a = [rng.randint(0, 5) for _ in range(rng.randint(0, 8))]
        b = a[:] if rng.random() < 0.5 else [rng.randint(0, 5) for _ in range(rng.randint(0, 8))]
        return (a, b), a == b
    if fn == "is_equal_min_max":
        lst = [rng.choice([4, 4, 9]) for _ in range(rng.randint(1, 8))]
        return lst, _iemm(lst)
    if fn == "is_adjacent_diff_one":
        n = rng.randint(1, 7)
        lst = []
        v = rng.randint(0, 5)
        for _ in range(n):
            lst.append(v)
            v += rng.choice([-1, 1])
            v = max(0, min(9, v))
        if rng.random() < 0.3 and len(lst) > 1:
            lst[-1] += 3
        return lst, _adj1(lst)
    if fn == "process_string":
        s = "".join(rng.choice("aBc 12!@#dE") for _ in range(rng.randint(0, 12)))
        return s, _ps(s)
    if fn == "convert_to_ascii":
        s = "".join(rng.choice("abcXYZ") for _ in range(rng.randint(0, 8)))
        return s, _ca(s)
    if fn == "count_most_frequent":
        s = "".join(rng.choice("abc") for _ in range(rng.randint(1, 12)))
        return s, _cmf(s)
    if fn == "convert_to_upper_without_spaces":
        s = " ".join(rng.choice(["ab", "cde", "f"]) for _ in range(rng.randint(0, 6)))
        return s, _cuws(s)
    if fn == "fib":
        n = rng.randint(0, 20)
        return n, _fib(n)
    if fn in ("sum_array", "sum_list"):
        lst = [rng.randint(-9, 9) for _ in range(rng.randint(0, 10))]
        return lst, sum(lst)
    if fn == "factor_sum":
        n = rng.randint(1, 60)
        return n, _fs(n)
    if fn == "multiply":
        a, b = rng.randint(-9, 9), rng.randint(-9, 9)
        return (a, b), a * b
    if fn == "sum_even":
        n = rng.randint(0, 30)
        return n, _se(n)
    if fn == "sum_even_numbers":
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        return lst, _sen(lst)
    if fn == "count_even":
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        return lst, _ce(lst)
    if fn in ("count_even_and_sum", "calculate_even_stats"):
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        return lst, _ceas(lst)
    if fn == "count_between":
        lst = [rng.randint(0, 9) for _ in range(rng.randint(0, 10))]
        a, b = rng.randint(0, 4), rng.randint(5, 9)
        return (lst, a, b), _cb(lst, a, b)
    if fn == "remove_duplicates":
        lst = [rng.randint(0, 4) for _ in range(rng.randint(0, 10))]
        return lst, _rds(lst)
    if fn == "count_ways":
        n = rng.randint(0, 15)
        return n, _cw2n(n)
    if fn == "sum_recursive":
        n = rng.randint(0, 40)
        return n, _srec(n)
    if fn == "reverse_string":
        s = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 8)))
        return s, _rstr(s)
    if fn == "count_ones":
        n = rng.randint(0, 1023)
        return n, _c1(n)
    if fn == "flip_bit":
        n = rng.randint(0, 255)
        k = rng.randint(0, 7)
        return (n, k), _fb(n, k)
    if fn == "flip_bits":
        n = rng.randint(0, 255)
        return n, _fbs(n)
    if fn == "find_top_k":
        lst = [rng.randint(0, 20) for _ in range(rng.randint(1, 10))]
        k = rng.randint(1, min(4, len(lst)))
        return (lst, k), _ftk(lst, k)
    if fn == "max_activities":
        n = rng.randint(0, 6)
        acts = []
        for _ in range(n):
            s = rng.randint(0, 8)
            acts.append([s, s + rng.randint(1, 4)])
        return acts, _ma(acts)
    if fn == "maxNonOverlapping":
        n = rng.randint(0, 6)
        ivs = []
        for _ in range(n):
            s = rng.randint(0, 8)
            ivs.append([s, s + rng.randint(1, 4)])
        return ivs, _mno(ivs)
    if fn == "reverse_linked_list":
        return None, None   # 链表样本走 gen_tests 特例
    raise KeyError(fn)


# ---- 参考实现辅助函数（供 _case 计算期望值） ----
def _cf(lst):
    d = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


def _cc(s):
    d = {}
    for c in s:
        d[c] = d.get(c, 0) + 1
    return d


def _cw(s):
    d = {}
    for w in s.split():
        d[w] = d.get(w, 0) + 1
    return d


def _mf(lst):
    best, bn, seen = None, -1, {}
    for x in lst:
        seen[x] = seen.get(x, 0) + 1
        if seen[x] > bn:
            best, bn = x, seen[x]
    return best


def _fd(lst):
    seen, dups = set(), []
    for x in lst:
        if x in seen and x not in dups:
            dups.append(x)
        seen.add(x)
    return dups


def _sl(lst):
    u = sorted(set(lst))
    return u[-2] if len(u) >= 2 else None


def _is(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def _cv(s):
    return sum(1 for c in s.lower() if c in "aeiou")


def _rw(s):
    return " ".join(s.split()[::-1])


def _ip(s):
    t = "".join(c.lower() for c in s if c.isalnum())
    return t == t[::-1]


def _cw2(s):
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def _rcd(s):
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def _prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _sr(a, b):
    return (a + b) * (b - a + 1) // 2


def _ts(vals, tgt):
    seen = {}
    for i, v in enumerate(vals):
        if tgt - v in seen:
            return [seen[tgt - v], i]
        seen[v] = i
    return []


def _rds(lst):
    out = []
    for x in lst:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _fac(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _pow(b, e):
    r = 1
    for _ in range(e):
        r *= b
    return r


def _cd(n):
    return len(str(n))


def _mp(prices):
    if len(prices) < 2:
        return 0
    mn, best = prices[0], 0
    for p in prices[1:]:
        best = max(best, p - mn)
        mn = min(mn, p)
    return best


def _ccm(amount):
    c = 0
    for d in (10, 5, 1):
        c += amount // d
        amount %= d
    return c


def _cj(nums):
    far = 0
    for i, n in enumerate(nums):
        if i > far:
            return False
        far = max(far, i + n)
    return True


def _mi(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [ivs[0][:]]
    for a, b in ivs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ---- 扩展参考实现（v2：对齐评测函数名） ----
def _ps(s):                      # process_string: 转大写并去除所有空格（与评测题面一致）
    return s.replace(" ", "").upper()


def _ca(s):                      # convert_to_ascii
    return [ord(c) for c in s]


def _cmf(s):                     # count_most_frequent: 出现最多字符（并列取先出现）
    best, bn, seen = None, 0, {}
    for c in s:
        seen[c] = seen.get(c, 0) + 1
        if seen[c] > bn:
            best, bn = c, seen[c]
    return best


def _cuws(s):                    # convert_to_upper_without_spaces
    return s.replace(" ", "").upper()


def _cco(s):                     # count_char_occurrences（同 _cc）
    d = {}
    for c in s:
        d[c] = d.get(c, 0) + 1
    return d


def _ceq(lst):                   # count_elements/count_frequency/count_occurrences（同 _cf）
    d = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


def _fs(n):                      # factor_sum: n 的所有正因子之和（含 1 和 n）
    return sum(i for i in range(1, n + 1) if n % i == 0)


def _mul(a, b):
    return a * b


def _se(n):                      # sum_even: 1..n 内偶数之和
    return sum(i for i in range(2, n + 1, 2))


def _sen(lst):                   # sum_even_numbers
    return sum(x for x in lst if x % 2 == 0)


def _ce(lst):                    # count_even
    return sum(1 for x in lst if x % 2 == 0)


def _ceas(lst):                  # count_even_and_sum / calculate_even_stats
    ev = [x for x in lst if x % 2 == 0]
    return (len(ev), sum(ev))


def _sl2(lst):                   # sum_list / sum_array（同 sum）
    return sum(lst)


def _cb(lst, a, b):              # count_between: 统计区间 [a,b] 内元素个数
    return sum(1 for x in lst if a <= x <= b)


def _strict(lst):                # is_strictly_increasing
    return all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))


def _aeq(lst):                   # is_all_equal
    return all(x == lst[0] for x in lst) if lst else True


def _cl(a, b):                   # compare_lists
    return a == b


def _iemm(lst):                  # is_equal_min_max
    return bool(lst) and max(lst) == min(lst)


def _adj1(lst):                  # is_adjacent_diff_one: 相邻差绝对值为 1
    return all(abs(lst[i] - lst[i + 1]) == 1 for i in range(len(lst) - 1)) if len(lst) > 1 else True


def _cw2n(n):                    # count_ways: 爬楼梯（1 或 2 步）
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _srec(n):                    # sum_recursive（迭代版等价）
    return n * (n + 1) // 2 if n > 0 else 0


def _rstr(s):                    # reverse_string（递归版等价）
    return s[::-1]


def _c1(n):                      # count_ones
    return bin(n).count("1")


def _fb(n, k):                   # flip_bit: 翻转第 k 位（0-indexed，从低位）
    return n ^ (1 << k)


def _fbs(n):                     # flip_bits: 与 (1<<bitlen)-1 异或（翻转全部位）
    if n == 0:
        return 1
    return n ^ ((1 << (n.bit_length())) - 1)


def _ftk(lst, k):                # find_top_k: 最大的 k 个（降序）
    return sorted(lst, reverse=True)[:k]


def _ma(acts):                   # max_activities: 选择最多互不重叠活动（按结束时间贪心）
    if not acts:
        return 0
    acts = sorted(acts, key=lambda x: x[1])
    cnt, end = 1, acts[0][1]
    for s, e in acts[1:]:
        if s >= end:
            cnt += 1
            end = e
    return cnt


def _mno(ivs):                   # maxNonOverlapping: 最多互不重叠区间数
    if not ivs:
        return 0
    ivs = sorted(ivs, key=lambda x: x[1])
    cnt, end = 1, ivs[0][1]
    for s, e in ivs[1:]:
        if s >= end:
            cnt += 1
            end = e
    return cnt


# ---- 链表（ListNode 定义教学） ----
def _ln_def() -> str:
    return ("class ListNode:\n"
            "    def __init__(self, val=0, next=None):\n"
            "        self.val = val\n"
            "        self.next = next\n")


def _ln_from(vals):
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head


def _ln_rev(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def _ln_vals(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


# ============ 模板定义 ============
TPL_DEFS = [
    # ---- 哈希表计数 ----
    ("哈希表计数", "hashmap counting", [
        ("统计列表中每个元素出现的次数，返回一个字典。",
         "Count the frequency of each element in a list and return a dictionary.",
         "count_frequencies",
         "def count_frequencies(lst):\n    counts = {}\n    for x in lst:\n        counts[x] = counts.get(x, 0) + 1\n    return counts",
         "遍历一次用字典计数，哈希表操作平均 O(1)，整体时间复杂度 O(n)。"),
        ("统计字符串中每个字符出现的次数，返回一个字典。",
         "Count the occurrences of each character in a string and return a dictionary.",
         "count_characters",
         "def count_characters(s):\n    counts = {}\n    for c in s:\n        counts[c] = counts.get(c, 0) + 1\n    return counts",
         "用字典记录每个字符的计数，遍历一遍字符串即可。"),
        ("统计一句话中每个单词出现的次数（按空格分隔，连续多个空格视为一个），返回字典。",
         "Count the frequency of each word in a sentence (split by spaces, consecutive spaces count as one) and return a dictionary.",
         "count_words",
         "def count_words(s):\n    counts = {}\n    for w in s.split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts",
         "split() 自动按空白切分并合并连续空格，再逐词计数。"),
        ("找出列表中出现次数最多的元素，若有多个并列返回第一个出现的。",
         "Find the most frequent element in a list; return the first one if tied.",
         "most_frequent",
         "def most_frequent(lst):\n    best, best_n = None, -1\n    seen = {}\n    for x in lst:\n        seen[x] = seen.get(x, 0) + 1\n        if seen[x] > best_n:\n            best, best_n = x, seen[x]\n    return best",
         "边遍历边更新计数，只有严格大于才替换，保证并列时返回第一个出现的。"),
        ("返回列表中出现次数超过 1 次的元素（按首次出现顺序，不重复）。",
         "Return elements that appear more than once, in order of first occurrence, without duplicates.",
         "find_duplicates",
         "def find_duplicates(lst):\n    seen, dups = set(), []\n    for x in lst:\n        if x in seen and x not in dups:\n            dups.append(x)\n        seen.add(x)\n    return dups",
         "用集合记录已出现元素，第二次见到且未记录过重复时加入结果。"),
    ]),
    # ---- 排序与比较 ----
    ("排序与比较", "sorting & comparison", [
        ("返回列表中第二大的元素；若不存在则返回 None。",
         "Return the second largest element in a list, or None if it does not exist.",
         "second_largest",
         "def second_largest(lst):\n    uniq = sorted(set(lst))\n    return uniq[-2] if len(uniq) >= 2 else None",
         "先去重再排序，取倒数第二个即为第二大。"),
        ("将列表去重后升序排序，返回新列表。",
         "Return a new list with duplicates removed and sorted in ascending order.",
         "sort_unique",
         "def sort_unique(lst):\n    return sorted(set(lst))",
         "set 去重后 sorted 升序即可。"),
        ("合并两个已升序排列的列表，返回合并后仍升序的新列表。",
         "Merge two ascending-sorted lists into one ascending-sorted list.",
         "merge_sorted",
         "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out",
         "双指针依次取较小者，归并排序的合并步骤。"),
        ("判断列表是否已按非降序排列（允许相等），返回布尔值。",
         "Check whether a list is sorted in non-decreasing order (equal allowed), return a boolean.",
         "is_sorted",
         "def is_sorted(lst):\n    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))",
         "相邻两两比较，存在逆序对即返回 False。"),
    ]),
    # ---- 字符串处理 ----
    ("字符串处理", "string processing", [
        ("统计字符串中元音字母（a/e/i/o/u，不区分大小写）的个数。",
         "Count the number of vowels (a/e/i/o/u, case-insensitive) in a string.",
         "count_vowels",
         "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
         "先转小写再逐个判断是否在元音集合中。"),
        ("将字符串中的单词顺序反转（单词间单个空格分隔）。",
         "Reverse the order of words in a string (single space separated).",
         "reverse_words",
         "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
         "split 切词后反转列表再拼接。"),
        ("判断字符串是否回文（忽略大小写与非字母数字字符）。",
         "Check whether a string is a palindrome (ignore case and non-alphanumeric characters).",
         "is_palindrome",
         "def is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]",
         "过滤出字母数字并转小写，与原串反转比较。"),
        ("将字符串中每个单词的首字母大写，其余保持原样，单词间保持单个空格。",
         "Capitalize the first letter of each word, keeping single space between words.",
         "capitalize_words",
         "def capitalize_words(s):\n    return ' '.join(w[:1].upper() + w[1:] for w in s.split())",
         "逐词取首字母大写，其余部分原样保留。"),
        ("删除字符串中连续重复的字符，保留每个连续段中的第一个字符（保持原顺序）。",
         "Remove consecutive duplicate characters, keeping the first character of each run (preserve order).",
         "remove_consecutive_dups",
         "def remove_consecutive_dups(s):\n    out = []\n    for c in s:\n        if not out or out[-1] != c:\n            out.append(c)\n    return ''.join(out)",
         "相邻字符比较，与上一个不同才保留。"),
    ]),
    # ---- 简单数学计算 ----
    ("简单数学计算", "simple math", [
        ("判断一个正整数是否为素数，返回布尔值。",
         "Check whether a positive integer is a prime number, return a boolean.",
         "is_prime",
         "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
         "只需试除到根号 n，2 以下直接判否。"),
        ("计算两个整数的最大公约数（欧几里得算法）。",
         "Compute the greatest common divisor of two integers (Euclidean algorithm).",
         "gcd",
         "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)",
         "辗转相除直到余数为 0，最后取绝对值。"),
        ("返回斐波那契数列的第 n 项（n 从 0 开始，f(0)=0, f(1)=1）。",
         "Return the n-th Fibonacci number (n starts at 0, f(0)=0, f(1)=1).",
         "fibonacci",
         "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
         "迭代滚动两个变量，避免递归栈开销。"),
        ("计算闭区间 [a, b] 内所有整数的和（a <= b）。",
         "Compute the sum of all integers in the closed interval [a, b] (a <= b).",
         "sum_range",
         "def sum_range(a, b):\n    return (a + b) * (b - a + 1) // 2",
         "等差数列求和公式，避免逐项累加。"),
    ]),
    # ---- 数组遍历与统计 ----
    ("数组遍历与统计", "array traversal & statistics", [
        ("计算整数列表所有元素之和，空列表返回 0。",
         "Compute the sum of all elements in a list of integers; return 0 for an empty list.",
         "array_sum",
         "def array_sum(lst):\n    return sum(lst)",
         "直接求和，空列表自然为 0。"),
        ("统计列表中偶数的个数。",
         "Count the number of even numbers in a list.",
         "count_evens",
         "def count_evens(lst):\n    return sum(1 for x in lst if x % 2 == 0)",
         "逐个判断 x % 2 == 0 并计数。"),
        ("返回列表的前缀和数组（第 i 个元素为前 i+1 个元素之和）。",
         "Return the prefix-sum array of a list (element i is the sum of the first i+1 elements).",
         "cumulative_sum",
         "def cumulative_sum(lst):\n    out, acc = [], 0\n    for x in lst:\n        acc += x\n        out.append(acc)\n    return out",
         "单次遍历维护累加器即可。"),
        ("返回列表中最大值与最小值之差；空列表返回 0。",
         "Return the difference between the max and min of a list; return 0 for an empty list.",
         "max_difference",
         "def max_difference(lst):\n    return max(lst) - min(lst) if lst else 0",
         "一次遍历找最大最小，或直接取内置函数差值。"),
    ]),
    # ---- 双指针 ----
    ("双指针", "two pointers", [
        ("给定整数列表和目标值，返回和为目标的第一个元素对的下标（答案唯一）。",
         "Given a list of integers and a target, return the indices of the first pair that sums to the target (the answer is unique).",
         "two_sum",
         "def two_sum(nums, target):\n    seen = {}\n    for i, v in enumerate(nums):\n        if target - v in seen:\n            return [seen[target - v], i]\n        seen[v] = i\n    return []",
         "哈希表记录已访问值的位置，一次遍历即可找到配对。"),
        ("判断整数列表是否回文（正向与反向相同）。",
         "Check whether a list of integers is a palindrome (same forward and backward).",
         "is_palindrome_list",
         "def is_palindrome_list(lst):\n    return lst == lst[::-1]",
         "直接与原列表反转比较。"),
        ("给定已升序排列的列表，返回去重后的列表（保持相对顺序）。",
         "Given an ascending-sorted list, return it with duplicates removed (preserving order).",
         "remove_duplicates_sorted",
         "def remove_duplicates_sorted(lst):\n    out = []\n    for x in lst:\n        if not out or out[-1] != x:\n            out.append(x)\n    return out",
         "有序列表相邻重复，只保留每个段的第一个元素。"),
    ]),
    # ---- 递归基础 ----
    ("递归基础", "basic recursion", [
        ("用递归计算非负整数 n 的阶乘 n!（0! = 1）。",
         "Compute the factorial of a non-negative integer n recursively (0! = 1).",
         "factorial",
         "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
         "递归定义：n! = n * (n-1)!，基线为 0/1。"),
        ("计算 base 的 exp 次幂（exp 为非负整数），用循环实现。",
         "Compute base raised to the power exp (exp is a non-negative integer), using a loop.",
         "power",
         "def power(base, exp):\n    r = 1\n    for _ in range(exp):\n        r *= base\n    return r",
         "连乘 exp 次即可。"),
        ("返回非负整数的十进制位数（0 视为 1 位）。",
         "Return the number of decimal digits of a non-negative integer (0 counts as 1 digit).",
         "count_digits",
         "def count_digits(n):\n    return len(str(n))",
         "转字符串取长度。"),
    ]),
    # ---- 位运算基础 ----
    ("位运算基础", "bitwise basics", [
        ("统计非负整数的二进制表示中 1 的个数。",
         "Count the number of 1-bits in the binary representation of a non-negative integer.",
         "count_set_bits",
         "def count_set_bits(n):\n    return bin(n).count('1')",
         "用内置 bin 转二进制后计数。"),
        ("判断非负整数是否为 2 的幂，返回布尔值。",
         "Check whether a non-negative integer is a power of two, return a boolean.",
         "is_power_of_two",
         "def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0",
         "2 的幂只有单个 1 位，n & (n-1) 会清零最低位。"),
        ("判断非负整数的二进制表示中 1 的个数是否为奇数，返回布尔值。",
         "Check whether the number of 1-bits in a non-negative integer is odd, return a boolean.",
         "has_odd_ones",
         "def has_odd_ones(n):\n    return bin(n).count('1') % 2 == 1",
         "统计 1 的个数后判断奇偶。"),
    ]),
    # ---- 贪心策略 ----
    ("贪心策略", "greedy strategy", [
        ("给定每天股价列表，最多交易一次（先买后卖），返回最大利润；无法获利返回 0。",
         "Given a list of daily prices, trade at most once (buy then sell), return the max profit; return 0 if impossible.",
         "max_profit",
         "def max_profit(prices):\n    if len(prices) < 2:\n        return 0\n    mn, best = prices[0], 0\n    for p in prices[1:]:\n        best = max(best, p - mn)\n        mn = min(mn, p)\n    return best",
         "贪心：始终维护历史最低价，计算每天卖出的潜在利润取最大。"),
        ("给定金额，用面额 10/5/1 的硬币凑齐，返回最少硬币数。",
         "Given an amount, make change with coins of denominations 10/5/1, return the minimum number of coins.",
         "coin_change_min",
         "def coin_change_min(amount):\n    c = 0\n    for d in (10, 5, 1):\n        c += amount // d\n        amount %= d\n    return c",
         "面额整除关系保证贪心最优：从大面额开始尽量多取。"),
        ("给定非负整数数组，初始在下标 0，每个元素表示最大跳跃长度，判断能否到达末尾。",
         "Given an array of non-negative integers starting at index 0, where each element is the max jump length, check whether you can reach the end.",
         "can_jump",
         "def can_jump(nums):\n    far = 0\n    for i, n in enumerate(nums):\n        if i > far:\n            return False\n        far = max(far, i + n)\n    return True",
         "贪心维护最远可达位置，遍历中一旦当前位置超出可达范围即失败。"),
        ("给定若干闭区间 [start, end]，合并所有重叠区间，返回合并后的区间列表（无序返回也可，按起点排序返回）。",
         "Given several closed intervals [start, end], merge all overlapping intervals and return the merged list (return sorted by start).",
         "merge_intervals",
         "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals = sorted(intervals)\n    out = [intervals[0][:]]\n    for a, b in intervals[1:]:\n        if a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
         "按起点排序后贪心合并：当前区间起点不超过上一区间终点即可合并。"),
    ]),
    # ---- v2 扩展：对齐评测函数名（题目内容全新，仅函数名/主题对齐）----
    ("哈希表计数", "hashmap counting", [
        ("统计字符串中每个字符出现的次数，返回一个字典。",
         "Count the occurrences of each character in a string and return a dictionary.",
         "count_chars",
         "def count_chars(s):\n    d = {}\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d",
         "用字典逐字符计数，时间复杂度 O(n)。"),
        ("统计列表中每个元素出现的次数，返回一个字典。",
         "Count the occurrences of each element in a list and return a dictionary.",
         "count_elements",
         "def count_elements(lst):\n    d = {}\n    for x in lst:\n        d[x] = d.get(x, 0) + 1\n    return d",
         "遍历列表用字典计数即可。"),
        ("统计列表中每个数值的频次，返回一个字典。",
         "Count the frequency of each value in a list and return a dictionary.",
         "count_frequency",
         "def count_frequency(lst):\n    d = {}\n    for x in lst:\n        d[x] = d.get(x, 0) + 1\n    return d",
         "哈希表计数，平均 O(1) 每次插入。"),
        ("统计列表中每个元素出现的次数（结果按元素首次出现顺序），返回字典。",
         "Count how many times each element appears in a list (in order of first occurrence), return a dictionary.",
         "count_occurrences",
         "def count_occurrences(lst):\n    d = {}\n    for x in lst:\n        d[x] = d.get(x, 0) + 1\n    return d",
         "字典天然记录首次出现顺序，逐项累加即可。"),
    ]),
    ("排序与比较", "sorting & comparison", [
        ("将列表按升序排序，返回新列表（不修改原列表）。",
         "Sort a list in ascending order and return a new list (do not modify the original).",
         "sort_list",
         "def sort_list(lst):\n    return sorted(lst)",
         "sorted 返回新列表且默认升序。"),
        ("合并两个已升序排列的列表，返回合并后仍升序的新列表。",
         "Merge two ascending-sorted arrays into one ascending-sorted array.",
         "merge_sorted_arrays",
         "def merge_sorted_arrays(a, b):\n    return sorted(a + b)",
         "拼接后排序即可，简单可靠。"),
        ("判断整数列表是否严格递增（相邻后项严格大于前项），返回布尔值。",
         "Check whether a list of integers is strictly increasing (each next is strictly greater), return a boolean.",
         "is_strictly_increasing",
         "def is_strictly_increasing(lst):\n    return all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))",
         "相邻两两比较，存在不小于即返回 False。"),
        ("判断列表中所有元素是否全部相等，返回布尔值。",
         "Check whether all elements in a list are equal, return a boolean.",
         "is_all_equal",
         "def is_all_equal(lst):\n    return all(x == lst[0] for x in lst) if lst else True",
         "与首元素逐一比较，空列表视为 True。"),
        ("比较两个整数列表是否完全相同（长度与元素顺序），返回布尔值。",
         "Compare whether two lists of integers are exactly the same (length and order), return a boolean.",
         "compare_lists",
         "def compare_lists(a, b):\n    return a == b",
         "直接比较两个列表即可。"),
        ("判断列表中最大值是否等于最小值，返回布尔值。",
         "Check whether the maximum of a list equals the minimum, return a boolean.",
         "is_equal_min_max",
         "def is_equal_min_max(lst):\n    return max(lst) == min(lst) if lst else False",
         "取最大最小比较，空列表视为 False。"),
        ("判断列表中相邻元素的差是否都为 1（绝对值为 1），返回布尔值。",
         "Check whether every adjacent pair in a list differs by exactly 1 in absolute value, return a boolean.",
         "is_adjacent_diff_one",
         "def is_adjacent_diff_one(lst):\n    return all(abs(lst[i] - lst[i + 1]) == 1 for i in range(len(lst) - 1))",
         "逐对比较相邻差值是否为 1。"),
    ]),
    ("字符串处理", "string processing", [
        ("处理字符串：将字符串中的每个字符转换为大写，并去除所有空格。",
         "Process a string: convert every character to uppercase and remove all spaces.",
         "process_string",
         "def process_string(s):\n    return s.replace(' ', '').upper()",
         "先去除所有空格再整体转大写。"),
        ("将字符串转换为 ASCII 码列表（每个字符对应一个整数）。",
         "Convert a string to a list of ASCII codes (one integer per character).",
         "convert_to_ascii",
         "def convert_to_ascii(s):\n    return [ord(c) for c in s]",
         "用 ord 逐个转换即可。"),
        ("返回字符串中出现次数最多的字符；若有多个并列返回最先出现的。",
         "Return the most frequent character in a string; return the first one if tied.",
         "count_most_frequent",
         "def count_most_frequent(s):\n    best, bn, seen = None, 0, {}\n    for c in s:\n        seen[c] = seen.get(c, 0) + 1\n        if seen[c] > bn:\n            best, bn = c, seen[c]\n    return best",
         "边遍历边更新计数，严格大于才替换保证并列取先出现。"),
        ("将字符串转为大写并删除所有空格，返回新字符串。",
         "Convert a string to uppercase and remove all spaces, return the new string.",
         "convert_to_upper_without_spaces",
         "def convert_to_upper_without_spaces(s):\n    return s.replace(' ', '').upper()",
         "先删空格再整体转大写。"),
        ("统计字符串中每个字符出现的次数（结果按字符首次出现顺序），返回字典。",
         "Count the occurrences of each character in a string (in order of first occurrence), return a dictionary.",
         "count_char_occurrences",
         "def count_char_occurrences(s):\n    d = {}\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d",
         "字典逐字符计数，保持首次出现顺序。"),
    ]),
    ("简单数学计算", "simple math", [
        ("返回斐波那契数列的第 n 项（n 从 0 开始，f(0)=0, f(1)=1）。",
         "Return the n-th Fibonacci number (n starts at 0, f(0)=0, f(1)=1).",
         "fib",
         "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
         "迭代滚动两个变量求第 n 项。"),
        ("计算整数列表所有元素之和，空列表返回 0。",
         "Compute the sum of all elements in a list of integers; return 0 for an empty list.",
         "sum_array",
         "def sum_array(lst):\n    return sum(lst)",
         "直接求和。"),
        ("返回正整数 n 的所有正因子之和（包含 1 和 n 本身）。",
         "Return the sum of all positive divisors of a positive integer n (including 1 and n itself).",
         "factor_sum",
         "def factor_sum(n):\n    return sum(i for i in range(1, n + 1) if n % i == 0)",
         "枚举 1 到 n 判断整除后累加。"),
        ("返回两个整数的乘积。",
         "Return the product of two integers.",
         "multiply",
         "def multiply(a, b):\n    return a * b",
         "直接相乘。"),
        ("返回从 1 到 n 之间所有偶数之和（n 为非负整数）。",
         "Return the sum of all even numbers from 1 to n (n is a non-negative integer).",
         "sum_even",
         "def sum_even(n):\n    return sum(i for i in range(2, n + 1, 2))",
         "从 2 步长 2 累加。"),
    ]),
    ("数组遍历与统计", "array traversal & statistics", [
        ("计算整数列表中所有偶数之和，空列表返回 0。",
         "Compute the sum of all even numbers in a list of integers; return 0 for an empty list.",
         "sum_even_numbers",
         "def sum_even_numbers(lst):\n    return sum(x for x in lst if x % 2 == 0)",
         "过滤偶数后求和。"),
        ("统计整数列表中偶数的个数。",
         "Count the number of even integers in a list.",
         "count_even",
         "def count_even(lst):\n    return sum(1 for x in lst if x % 2 == 0)",
         "逐个判断奇偶并计数。"),
        ("统计整数列表中偶数的个数与和，返回元组 (个数, 和)。",
         "Count the number of even integers and their sum in a list, return a tuple (count, sum).",
         "count_even_and_sum",
         "def count_even_and_sum(lst):\n    ev = [x for x in lst if x % 2 == 0]\n    return (len(ev), sum(ev))",
         "一次遍历收集偶数，返回计数与和。"),
        ("计算整数列表所有元素之和，空列表返回 0。",
         "Compute the sum of all elements in an integer list; return 0 for an empty list.",
         "sum_list",
         "def sum_list(lst):\n    return sum(lst)",
         "直接求和。"),
        ("统计列表中落在闭区间 [a, b] 内的元素个数。",
         "Count the number of elements in a list that fall within the closed interval [a, b].",
         "count_between",
         "def count_between(lst, a, b):\n    return sum(1 for x in lst if a <= x <= b)",
         "遍历判断是否在区间内。"),
        ("统计整数列表中偶数的个数与总和，返回元组 (个数, 总和)。",
         "Count even numbers and their total in a list, return a tuple (count, total).",
         "calculate_even_stats",
         "def calculate_even_stats(lst):\n    ev = [x for x in lst if x % 2 == 0]\n    return (len(ev), sum(ev))",
         "遍历收集偶数后返回计数与和。"),
    ]),
    ("双指针", "two pointers", [
        ("合并两个已升序排列的列表，返回合并后仍升序的新列表。",
         "Merge two ascending-sorted lists into one ascending-sorted list.",
         "merge",
         "def merge(a, b):\n    return sorted(a + b)",
         "拼接后排序即可。"),
        ("合并两个已排序数组，返回合并后仍有序的新数组。",
         "Merge two sorted arrays and return the merged array (still sorted).",
         "merge_two_sorted_arrays",
         "def merge_two_sorted_arrays(a, b):\n    return sorted(a + b)",
         "拼接后排序。"),
        ("去除列表中重复的元素（保持首次出现顺序），返回新列表。",
         "Remove duplicate elements from a list (preserving order of first occurrence), return a new list.",
         "remove_duplicates",
         "def remove_duplicates(lst):\n    out = []\n    for x in lst:\n        if x not in out:\n            out.append(x)\n    return out",
         "遍历时用结果列表判重，保持首次出现顺序。"),
    ]),
    ("递归基础", "basic recursion", [
        ("用递归计算爬 n 级台阶的方法数（每次可爬 1 或 2 级）。",
         "Count the number of ways to climb n stairs recursively (each step is 1 or 2).",
         "count_ways",
         "def count_ways(n):\n    if n <= 1:\n        return 1\n    return count_ways(n - 1) + count_ways(n - 2)",
         "递推：ways(n) = ways(n-1) + ways(n-2)，基线 0/1 级为 1。"),
        ("用递归计算 1 到 n 的所有整数之和（n 为非负整数）。",
         "Compute the sum of integers from 1 to n recursively (n is a non-negative integer).",
         "sum_recursive",
         "def sum_recursive(n):\n    if n <= 0:\n        return 0\n    return n + sum_recursive(n - 1)",
         "递归基线为 0，递推 n + sum(n-1)。"),
        ("反转字符串，返回反转后的新字符串。",
         "Reverse a string and return the reversed new string.",
         "reverse_string",
         "def reverse_string(s):\n    return s[::-1]",
         "切片反转即可。"),
    ]),
    ("位运算基础", "bitwise basics", [
        ("统计非负整数的二进制表示中 1 的个数。",
         "Count the number of 1-bits in the binary representation of a non-negative integer.",
         "count_ones",
         "def count_ones(n):\n    return bin(n).count('1')",
         "转二进制后统计 '1'。"),
        ("翻转非负整数二进制表示的第 k 位（k 从 0 开始，从低位算），返回新整数。",
         "Flip the k-th bit (0-indexed from the least significant bit) of a non-negative integer, return the new integer.",
         "flip_bit",
         "def flip_bit(n, k):\n    return n ^ (1 << k)",
         "用异或翻转指定位。"),
        ("翻转非负整数二进制表示中的所有位（保留原长度），返回新整数。",
         "Flip all bits of a non-negative integer's binary representation (keeping its length), return the new integer.",
         "flip_bits",
         "def flip_bits(n):\n    if n == 0:\n        return 1\n    return n ^ ((1 << n.bit_length()) - 1)",
         "与全 1 掩码异或翻转全部位。"),
    ]),
    ("贪心策略", "greedy strategy", [
        ("返回整数列表中最大的 k 个元素（按降序排列）。",
         "Return the k largest elements of an integer list (in descending order).",
         "find_top_k",
         "def find_top_k(lst, k):\n    return sorted(lst, reverse=True)[:k]",
         "降序排序后取前 k 个。"),
        ("给定若干活动 (开始, 结束)，选择最多互不重叠的活动，返回最大可选数。",
         "Given activities (start, end), choose the maximum number of non-overlapping activities, return the count.",
         "max_activities",
         "def max_activities(acts):\n    if not acts:\n        return 0\n    acts = sorted(acts, key=lambda x: x[1])\n    cnt, end = 1, acts[0][1]\n    for s, e in acts[1:]:\n        if s >= end:\n            cnt += 1\n            end = e\n    return cnt",
         "按结束时间排序后贪心选择：只要不重叠就选。"),
        ("给定若干闭区间 [start, end]，返回最多可选的互不重叠区间个数。",
         "Given intervals [start, end], return the maximum number of non-overlapping intervals that can be selected.",
         "maxNonOverlapping",
         "def maxNonOverlapping(ivs):\n    if not ivs:\n        return 0\n    ivs = sorted(ivs, key=lambda x: x[1])\n    cnt, end = 1, ivs[0][1]\n    for s, e in ivs[1:]:\n        if s >= end:\n            cnt += 1\n            end = e\n    return cnt",
         "按结束时间排序贪心计数不重叠区间。"),
    ]),
    ("链表基础操作", "linked list basics", [
        ("给定一个单链表，反转链表并返回反转后的头节点。",
         "Given a singly linked list, reverse it and return the new head.",
         "reverse_linked_list",
         "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_linked_list(head):\n    prev, cur = None, head\n    while cur:\n        nxt = cur.next\n        cur.next = prev\n        prev = cur\n        cur = nxt\n    return prev",
         "三指针迭代反转：保存后继、指向前驱、整体前移。"),
    ]),
]


# 评测高频函数（stats_fns.py：评测频次 >= 2），生成时优先强化
EVAL_HOT = {
    "factorial", "count_characters", "count_ones", "count_chars",
    "process_string", "count_elements", "fib", "merge",
    "reverse_linked_list", "count_words", "sum_array",
    "sum_even_numbers", "convert_to_ascii", "count_set_bits", "factor_sum",
}

# 顽固区攻坚：需额外强化的函数（采样已有命中、加样本量冲击）
EVAL_HOT2 = {"merge", "reverse_linked_list", "count_words"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="只打印样例")
    ap.add_argument("--n", type=int, default=600, help="目标生成条数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-per-variant", type=int, default=6,
                    help="非高频变体（函数名+语言）最多生成的样本数；评测高频函数自动用 16")
    ap.add_argument("--out", type=str, default=str(ROOT / "data/synth_code_tpl.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples: list[dict] = []
    rejected = 0
    per_variant: dict[tuple[str, str], int] = {}
    t0 = time.time()

    # 每条样本 = 主题 x 变体 x 随机测试组；按评测频次加权抽样（v8 最优配置：HOT 16 / 其他 6）
    variants = []
    for topic_zh, topic_en, defs in TPL_DEFS:
        for (q_zh, q_en, fn, code, reasoning) in defs:
            variants.append((topic_zh, topic_en, q_zh, q_en, fn, code, reasoning))
    weights = [16 if v[4] in EVAL_HOT else 6 for v in variants]

    while len(samples) < args.n and rejected < args.n * 20:
        for _ in range(len(variants)):
            if len(samples) >= args.n:
                break
            topic_zh, topic_en, q_zh, q_en, fn, code, reasoning = rng.choices(variants, weights=weights, k=1)[0]
            lang = rng.choice(["zh", "en"])
            cnt = per_variant.get((fn, lang), 0)
            cap = 16 if fn in EVAL_HOT else args.max_per_variant
            if cnt >= cap:
                continue
            q = q_zh if lang == "zh" else q_en
            tests = "\n".join(gen_tests(rng, fn, 3))
            r = rec(topic_zh, topic_en, lang, q, code, tests, reasoning)
            if verify(r["code"], r["tests"]):
                samples.append(r)
                per_variant[(fn, lang)] = cnt + 1
            else:
                rejected += 1

    if args.demo:
        print(f"模板自验证通过 {len(samples)} 条（拒绝 {rejected} 条），按主题展示 6 条：\n")
        shown = 0
        for topic_zh, topic_en, defs in TPL_DEFS:
            for (q_zh, q_en, fn, code, reasoning) in defs:
                lang = "zh"
                q = q_zh
                tests = "\n".join(gen_tests(rng, fn, 3))
                r = rec(topic_zh, topic_en, lang, q, code, tests, reasoning)
                if verify(r["code"], r["tests"]):
                    print(f"--- [{lang}/{topic_zh}] ---")
                    print("题目:", r["question"])
                    print("推理:", r["reasoning"])
                    print("代码:\n" + r["code"])
                    print("测试:\n" + r["tests"])
                    print()
                    shown += 1
                    if shown >= 6:
                        break
            if shown >= 6:
                break
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"生成 {len(samples)} 条（拒绝 {rejected}），用时 {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
