"""数据源探针 v2：结果写日志文件，避免管道缓冲丢输出。"""
import os, sys, time

LOG = open(r"F:\自研大模型\logs\probe_data.log", "w", encoding="utf-8", buffering=1)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def test_ms():
    """ModelScope 国内源，最稳。"""
    log("\n########## ModelScope ##########")
    try:
        import modelscope
        log("modelscope version:", modelscope.__version__)
    except Exception as e:
        log("modelscope NOT installed:", type(e).__name__, str(e)[:150])
        return
    from modelscope.msdatasets import MsDataset
    for name, ns, sub in [
        ("ms-cn-wiki", "sentence-transformers/wikipedia-cn", "corpus"),
        ("ms-wikipedia-zh", "iic/ wikipedia_zh", None),
    ]:
        pass
    # 直接试 MsDataset 加载
    try:
        t = time.time()
        ds = MsDataset.load(
            "sentence-transformers/wikipedia-cn",
            split="train",
            subset_name="corpus",
            trust_remote_code=True,
        )
        row = ds[0]
        log(f"[OK] modelscope wikipedia-cn {time.time()-t:.1f}s keys={list(row.keys())}")
        log("     sample:", str(row.get("text", row.get("title", "")))[:150].replace("\n", " "))
    except Exception as e:
        log("[FAIL] modelscope:", type(e).__name__, str(e)[:300])


def test_hf(repo, config, label):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    log(f"\n----- HF {label}: {repo} / {config} -----")
    t = time.time()
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, config, split="train", streaming=True)
        row = next(iter(ds))
        txt = ""
        for k in ("text", "content", "question"):
            if k in row:
                txt = str(row[k])[:120].replace("\n", " ")
                break
        log(f"[OK] {label} {time.time()-t:.1f}s keys={list(row.keys())}")
        log("     sample:", txt)
    except Exception as e:
        log(f"[FAIL] {label} {time.time()-t:.1f}s {type(e).__name__}: {str(e)[:250]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ms"):
        test_ms()
    if which in ("all", "hf"):
        test_hf("wikimedia/wikipedia", "20231101.zh", "zhwiki")
        test_hf("openai/gsm8k", "main", "gsm8k")
    log("\n=== DONE ===")
    LOG.close()