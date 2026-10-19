"""数据通路探针：验证哪些语料源在本机网络下可用（只取首样本，不整包下载）。"""
import os, sys, time, traceback

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import load_dataset

CANDIDATES = [
    ("zh-wiki", "wikimedia/wikipedia", "20231101.zh"),
    ("zh-wiki-lite", "wikimedia/wikipedia", "20220301.zh"),
    ("en-fineweb", "HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ("gsm8k", "openai/gsm8k", "main"),
]


def probe(name, repo, config):
    t = time.time()
    try:
        ds = load_dataset(repo, config, split="train", streaming=True)
        row = next(iter(ds))
        keys = list(row.keys())
        sample = ""
        for k in ("text", "content", "question", "answer"):
            if k in row:
                sample = str(row[k])[:100].replace("\n", " ")
                break
        print(f"[OK]   {name:14s} {time.time()-t:6.1f}s keys={keys}")
        print(f"       sample: {sample}")
        return True
    except Exception as e:
        print(f"[FAIL] {name:14s} {time.time()-t:6.1f}s {type(e).__name__}: {str(e)[:200]}")
        return False


if __name__ == "__main__":
    ok = []
    for name, repo, config in CANDIDATES:
        try:
            ok.append((name, probe(name, repo, config)))
        except KeyboardInterrupt:
            break
    print("\n=== SUMMARY ===")
    for name, good in ok:
        print(f"  {name}: {'USABLE' if good else 'NO'}")