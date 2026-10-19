"""Inspect the recovered checkpoint's actual config and model tensor structure."""
import json
import torch
import torch.serialization as ts
import torch._utils
import pickle
import io
from pathlib import Path

REC = Path(r"F:\自研大模型\runs\a003_p_bot_s2\last_recovered.pt")
reader = torch._C.PyTorchFileReader(open(REC, "rb"))
pkl_bytes = None
for cand in ("data.pkl", "archive/data.pkl"):
    try:
        pkl_bytes = reader.get_record(cand)
        print("pickle record:", cand, len(pkl_bytes), "bytes")
        break
    except Exception:
        continue
assert pkl_bytes is not None

loaded = {}


def get_storage(key, nbytes):
    try:
        return (
            reader.get_storage_from_record(f"data/{key}", nbytes, torch.UntypedStorage)
            ._typed_storage()
            ._untyped_storage
        )
    except Exception:
        return torch.empty(nbytes, dtype=torch.uint8).untyped_storage()


def persistent_load(saved_id):
    typename = saved_id[0].decode() if isinstance(saved_id[0], bytes) else saved_id[0]
    storage_type, key, location, numel = saved_id[1:]
    dtype = torch.uint8 if storage_type is torch.UntypedStorage else storage_type.dtype
    if key in loaded:
        return loaded[key]
    nbytes = numel * torch._utils._element_size(dtype)
    st = get_storage(key, nbytes)
    typed = torch.storage.TypedStorage(wrap_storage=st, dtype=dtype, _internal=True)
    loaded[key] = typed
    return typed


class U(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return ts.StorageType(name)
            except KeyError:
                pass
        return super().find_class(mod_name, name)


ts._serialization_tls.map_location = "cpu"
up = U(io.BytesIO(pkl_bytes))
up.persistent_load = persistent_load
ck = up.load()
ts._serialization_tls.map_location = None

print("=== checkpoint step ===")
print("step:", ck["step"])

print("=== actual training config (from checkpoint) ===")
cfg = ck["config"]
for k, v in cfg.items():
    print(f"  {k}: {v}")

model = ck["model"]
print("=== model state_dict: %d keys ===" % len(model))
total = 0
emb_like = []
for k, t in model.items():
    n = t.numel()
    total += n
    if n == 12582912:
        emb_like.append(k)
print("total model params:", total)
print("embedding-sized tensors (12582912):", emb_like)
zero = [k for k, t in model.items() if t.numel() and not bool(t.abs().sum().item())]
print("all-zero model tensors:", zero if zero else "none")

print("=== loss_curve n_params (record field) ===")
lc = json.loads(
    Path(r"F:\自研大模型\runs\a003_p_bot_s2\loss_curve.json").read_text(encoding="utf-8")
)
print("p_bot_s2 loss_curve n_params:", lc.get("n_params"))
try:
    lc_mid = json.loads(
        Path(r"F:\自研大模型\runs\a003_p_mid\loss_curve.json").read_text(encoding="utf-8")
    )
    print("p_mid loss_curve n_params:", lc_mid.get("n_params"))
except Exception as e:
    print("p_mid loss_curve read failed:", e)

print("=== baseline runs/pretrain final.pt param count ===")
b = torch.load(
    Path(r"F:\自研大模型\runs\pretrain\final.pt"), map_location="cpu", weights_only=True
)
bn = sum(t.numel() for t in b["model"].values())
print("baseline pretrain params:", bn)
print("baseline state_dict keys:", len(b["model"]))
bzero = [k for k, t in b["model"].items() if t.numel() and not bool(t.abs().sum().item())]
print("baseline all-zero tensors:", bzero if bzero else "none")
