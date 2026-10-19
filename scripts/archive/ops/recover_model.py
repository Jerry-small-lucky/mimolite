"""Recover the MODEL weights from the truncated a003_p_bot_s2 last.pt.

The zip on disk contains tensor records data/0..data/297; the pickle inside
references further tensors (data/298+, all optimizer state) that were never
written. We unpickle with a tolerant persistent_load: present storages come
from the recovered zip, missing ones are zero-filled. Only the 'model'
sub-dict is needed for final.pt.

Verify: weights_only=True load of final.pt, param count == 108,160,896,
and no model tensor is zero-filled garbage.
"""
import io
import os
import time
import pickle
import torch
import torch._utils
import torch.serialization as ts
from pathlib import Path

REC = Path(r"F:\自研大模型\runs\a003_p_bot_s2\last_recovered.pt")
FINAL = Path(r"F:\自研大模型\runs\a003_p_bot_s2\final.pt")
# True param count (embed.weight + lm_head.weight both in state_dict despite
# tie_embeddings=True). Cross-checked against runs/pretrain/final.pt:
# baseline = 120,743,808, 170 keys, no zero tensors.
EXPECTED = 120743808

for attempt in range(3):
    if REC.exists():
        break
    time.sleep(2)
assert REC.exists(), f"missing {REC}"

reader = torch._C.PyTorchFileReader(open(REC, "rb"))

# --- pickle bytes ---
pkl_name = None
for cand in ("archive/data.pkl", "data.pkl"):
    try:
        pkl_bytes = reader.get_record(cand)
        pkl_name = cand
        break
    except Exception:
        continue
assert pkl_name is not None, "cannot locate pickle record"
print("pickle record:", pkl_name, len(pkl_bytes), "bytes")

loaded_storages: dict = {}
zero_filled: list = []


def get_storage(key, nbytes):
    try:
        return reader.get_storage_from_record(
            f"data/{key}", nbytes, torch.UntypedStorage
        )._typed_storage()._untyped_storage
    except Exception:
        raw = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
        zero_filled.append(key)
        return raw.untyped_storage()


def persistent_load(saved_id):
    typename = saved_id[0].decode() if isinstance(saved_id[0], bytes) else saved_id[0]
    if typename != "storage":
        raise AssertionError(f"unknown typename {typename}")
    storage_type, key, location, numel = saved_id[1:]
    dtype = torch.uint8 if storage_type is torch.UntypedStorage else storage_type.dtype
    if key in loaded_storages:
        return loaded_storages[key]
    nbytes = numel * torch._utils._element_size(dtype)
    storage = get_storage(key, nbytes)
    typed = torch.storage.TypedStorage(
        wrap_storage=storage, dtype=dtype, _internal=True
    )
    loaded_storages[key] = typed
    return typed


class RecoverUnpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return ts.StorageType(name)
            except KeyError:
                pass
        mod_name = {"torch.tensor": "torch._tensor"}.get(mod_name, mod_name)
        return super().find_class(mod_name, name)


ts._serialization_tls.map_location = "cpu"
try:
    up = RecoverUnpickler(io.BytesIO(pkl_bytes))
    up.persistent_load = persistent_load
    ck = up.load()
finally:
    ts._serialization_tls.map_location = None

print("unpickled keys:", list(ck.keys()))
print("zero-filled storage keys:", len(zero_filled), "->", zero_filled[:10],
      "..." if len(zero_filled) > 10 else "")
print("total storages loaded:", len(loaded_storages))

model = ck["model"]
n = sum(t.numel() for t in model.values())
print("model params:", n, "| expected", EXPECTED)
assert n == EXPECTED, "param count mismatch"
# sanity: no all-zero model tensor (would indicate an accidentally zero-filled
# tensor that actually belongs to model)
bad = [k for k, t in model.items() if t.numel() and not bool(t.abs().sum().item())]
print("all-zero model tensors:", bad if bad else "none")
assert not bad

final = {
    "model": model,
    "step": ck["step"],
    "config": ck["config"],
    "tokenizer_meta": ck["tokenizer_meta"],
    "hist": ck.get("hist", [])[-2000:],
}
torch.save(final, FINAL)

chk = torch.load(FINAL, map_location="cpu", weights_only=True)
n2 = sum(t.numel() for t in chk["model"].values())
print("final.pt verified: step", chk["step"], "| params", n2,
      "| size", os.path.getsize(FINAL))
assert n2 == EXPECTED and chk["step"] == 1200
print("RECOVERY OK")
