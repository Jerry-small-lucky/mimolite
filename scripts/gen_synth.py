"""
蒸馏合成语料 —— 用 ollama qwen3-1.7b 把维基知识段落改写为讲义风格
==================================================================
模式 A1：段落改写为教科书讲义风格（条理化、通俗、教学语气）
模式 A2：段落提炼 3-5 句核心要点
中英按 75:25（zh_ratio=0.75，与训练混合一致）。预算默认 3M token，
按字符/token 估算（zh 1.85 字/token，en 2.33 字符/token）累计。
断点续跑：progress 文件记录游标与累计；输出增量写盘。
质量护栏：过短输出丢弃；含拒绝词/考题词丢弃（C-Eval 零泄漏）。
单线程串行请求 ollama，日志自 flush。
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根
DATA = ROOT / "data"
LOG = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3-1.7b"
ZH_RATIO = 0.75
BUDGET_TOKENS = 5_000_000
PROG = DATA / "synth_progress.json"

ZH_REWRITE = (
    "你是一名知识科普作者。请把下面这段知识文本改写成教科书讲义风格："
    "事实必须准确，条理清晰，语言通俗易懂，可适当补充必要的衔接和解释，"
    "不要添加与原文无关的新事实，不要输出选择题、考试题或评测题，"
    "直接输出改写后的正文，不要输出思考过程。\n\n原文：\n{text}"
)
EN_REWRITE = (
    "You are an educational writer. Rewrite the following knowledge passage "
    "in a textbook-lecture style: keep facts accurate, organize ideas clearly, "
    "use plain language, do NOT add unrelated facts, do NOT produce quiz, exam "
    "or test questions, output only the rewritten passage without thinking text.\n\n"
    "Source:\n{text}"
)
ZH_DIGEST = (
    "你是一名知识科普作者。请把下面这段知识文本提炼成 3-5 句核心要点："
    "只保留最重要的事实，句子通顺，不要输出选择题、考试题或评测题，"
    "直接输出要点，不要输出思考过程。\n\n原文：\n{text}"
)
EN_DIGEST = (
    "You are an educational writer. Summarize the following knowledge passage "
    "into 3-5 key sentences with the most important facts only. Do NOT produce "
    "quiz, exam or test questions. Output only the summary without thinking text.\n\n"
    "Source:\n{text}"
)

REJECT = (
    "抱歉", "对不起", "我不确定", "无法", "不能", "作为AI", "作为 AI",
    "语言模型", "sorry", "I cannot", "I don't know", "as an AI", "AI assistant",
    "选择题", "考试", "评测题", "测试题", "quiz", "exam question",
)
REJECT_DOT = (".", "。", "!", "！", "?", "？", ":", "：", "，", ",", " ")  # 见 _strip_think
MIN_OUT_CHARS = {"zh": 60, "en": 80}


def _strip_think(out: str) -> str:
    """qwen3 带 <think> 推理：丢弃整段思考痕迹，截取正文输出。"""
    if "<think>" in out:
        out = out.split("<think>", 1)[1]
        if "</think>" in out:
            out = out.split("</think>", 1)[1]
        else:
            # 未闭合：从「最后一段正文」切（启发式：取末段）
            out = out.rsplit("\n", 1)[-1]
    return out.strip()


def _clean_out(out: str) -> str:
    out = _strip_think(out)
    # 去掉可能残留的「好的/让我/下面/我将」开场白
    for lead in ("好的，", "好的,", "好的 ", "好的", "让我来", "我来", "下面", "我将",
                 "OK,", "Sure,", "Alright,", "Here is", "Here's", "Let me"):
        if out.startswith(lead):
            out = out[len(lead):].lstrip("，,。:： ")
            break
    return out.strip()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOG / "synth.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def call_ollama(prompt: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 512, "temperature": 0.7, "top_p": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["response"]


def est_tokens(text: str, lang: str) -> int:
    return max(1, int(len(text) / (1.85 if lang == "zh" else 2.33)))


def load_segments(lang: str):
    """优先用清洗后语料，回退到原语料。"""
    for name in (f"clean_{lang}.jsonl", f"pretrain_{lang}.orig.jsonl"):
        p = DATA / name
        if p.exists():
            segs = [json.loads(l)["text"] for l in p.open(encoding="utf-8")]
            log(f"{lang}: 采样源 {name}，{len(segs):,} 段")
            return segs
    return []


def main() -> None:
    budget_zh = int(BUDGET_TOKENS * ZH_RATIO)
    budget_en = BUDGET_TOKENS - budget_zh

    zh = load_segments("zh")
    en = load_segments("en")
    random.seed(42)
    random.shuffle(zh)
    random.shuffle(en)

    prog = {"zh_idx": 0, "en_idx": 0, "zh_tok": 0, "en_tok": 0,
            "zh_n": 0, "en_n": 0, "start": time.time()}
    if PROG.exists():
        prog.update(json.loads(PROG.read_text(encoding="utf-8")))
        log(f"断点续跑：zh {prog['zh_tok']:,}/{budget_zh:,} tok, "
            f"en {prog['en_tok']:,}/{budget_en:,} tok")

    out_zh = (DATA / "hq_synth_zh.jsonl").open("a", encoding="utf-8")
    out_en = (DATA / "hq_synth_en.jsonl").open("a", encoding="utf-8")

    def save_prog():
        PROG.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")

    try:
        while (prog["zh_tok"] < budget_zh or prog["en_tok"] < budget_en) and (
            prog["zh_idx"] < len(zh) or prog["en_idx"] < len(en)
        ):
            # 语言轮换调度：zh 优先，保持 75:25
            use_zh = (
                prog["zh_tok"] < budget_zh
                and (prog["en_tok"] >= budget_en
                     or prog["zh_tok"] / max(1, prog["zh_tok"] + prog["en_tok"]) < ZH_RATIO)
            )
            if use_zh:
                if prog["zh_idx"] >= len(zh):
                    break
                text = zh[prog["zh_idx"]]
                lang, budget = "zh", budget_zh
            else:
                if prog["en_idx"] >= len(en):
                    break
                text = en[prog["en_idx"]]
                lang, budget = "en", budget_en

            digest = (prog[f"{lang}_n"] % 3) == 2  # 每 3 条 1 条提炼
            prompt = (ZH_DIGEST if digest else ZH_REWRITE).format(text=text[:900]) \
                if lang == "zh" else \
                (EN_DIGEST if digest else EN_REWRITE).format(text=text[:1200])
            try:
                out = _clean_out(call_ollama(prompt))
            except Exception as e:
                log(f"请求失败: {e}，暂停 20s")
                time.sleep(20)
                continue

            ok = len(out) >= MIN_OUT_CHARS[lang] and not any(
                w in out for w in REJECT
            )
            if ok:
                (out_zh if lang == "zh" else out_en).write(
                    json.dumps({"text": out, "lang": lang}, ensure_ascii=False) + "\n"
                )
                tok = est_tokens(out, lang)
                prog[f"{lang}_tok"] += tok
                prog[f"{lang}_n"] += 1
            else:
                tok = 0
            prog[f"{lang}_idx"] += 1

            if (prog["zh_n"] + prog["en_n"]) % 20 == 0:
                save_prog()
                el = (time.time() - prog["start"]) / 60
                rate = (prog["zh_tok"] + prog["en_tok"]) / max(1, el) / 60
                log(
                    f"zh {prog['zh_tok']:,}/{budget_zh:,} ({prog['zh_n']}条) "
                    f"en {prog['en_tok']:,}/{budget_en:,} ({prog['en_n']}条) "
                    f"| {rate:.0f} tok/s | 已用 {el:.0f} min"
                )
    finally:
        save_prog()
        out_zh.close()
        out_en.close()

    log(
        f"完成：zh {prog['zh_tok']:,} tok / {prog['zh_n']:,} 条；"
        f"en {prog['en_tok']:,} tok / {prog['en_n']:,} 条"
    )


if __name__ == "__main__":
    main()
