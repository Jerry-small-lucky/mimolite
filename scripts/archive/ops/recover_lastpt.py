"""Full recovery of the truncated a003_p_bot_s2 last.pt.

Strategy:
- 301 entries resolve cleanly via data descriptors (CRC-verified).
- data/297 fills the file exactly to EOF (6,291,456 B, a complete fp32
  tensor whose data was flushed before the descriptor/version records were
  written). Its expected size is cross-checked by torch.load: on storage
  size mismatch we drop it and retry.
- torch 2.11's PyTorchFileReader requires a 'version' record (and a
  .data/serialization_id); both are copied byte-for-byte from a tiny
  torch.save made in the same environment, under both 'version' and
  'archive/version' names.
"""
import struct
import os
import io
import zipfile
import zlib
import tempfile
import torch
from pathlib import Path

SRC = Path(r"F:\自研大模型\runs\a003_p_bot_s2\last.pt")
REC = Path(r"F:\自研大模型\runs\a003_p_bot_s2\last_recovered.pt")
FINAL = Path(r"F:\自研大模型\runs\a003_p_bot_s2\final.pt")


def local_headers(data: bytes):
    offs = []
    i = 0
    while True:
        i = data.find(b"PK\x03\x04", i)
        if i < 0:
            break
        offs.append(i)
        i += 4
    return offs


def parse(data: bytes, offs):
    fsize = len(data)
    entries = []
    for idx, off in enumerate(offs):
        sig, ver, flags, method, mt, md, crc, csize, usize, nl, el = struct.unpack(
            "<IHHHHHIIIHH", data[off : off + 30]
        )
        name = data[off + 30 : off + 30 + nl].decode("utf-8", "replace")
        ds = off + 30 + nl + el
        end = None
        if flags & 0x08:
            for j in range(idx + 1, len(offs) + 1):
                nxt = offs[j] if j < len(offs) else fsize
                for dl in (16, 12):
                    if nxt - dl < ds:
                        continue
                    if dl == 16:
                        d_sig, d_crc, d_csize, d_usize = struct.unpack(
                            "<IIII", data[nxt - dl : nxt]
                        )
                        if d_sig != 0x08074B50:
                            continue
                    else:
                        d_crc, d_csize, d_usize = struct.unpack(
                            "<III", data[nxt - dl : nxt]
                        )
                    if d_csize == nxt - dl - ds and (
                        zlib.crc32(data[ds : nxt - dl]) & 0xFFFFFFFF
                    ) == d_crc:
                        end, crc, usize, csize = nxt - dl, d_crc, d_usize, d_csize
                        break
                if end is not None:
                    break
        else:
            end = ds + csize
        if end is None:
            continue
        entries.append([name, crc, usize, ds, end])
    return entries


def tiny_records():
    """Copy 'version' and serialization_id bytes from a torch.save in this env."""
    tmp = os.path.join(tempfile.gettempdir(), "_layout_probe.pt")
    torch.save({"probe": torch.zeros(1)}, tmp)
    got = {}
    with zipfile.ZipFile(tmp) as z:
        for n in z.namelist():
            if n == "version" or n.endswith("/version"):
                got.setdefault("version", z.read(n))
            if n == ".data/serialization_id" or n.endswith("/.data/serialization_id"):
                got.setdefault("serialization_id", z.read(n))
    os.remove(tmp)
    return got


def rebuild(entries, extra, out_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, crc, usize, ds, end in entries:
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, data[ds:end])
        for name, blob in extra.items():
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, blob)
    open(out_path, "wb").write(buf.getvalue())


data = open(SRC, "rb").read()
offs = local_headers(data)
entries = parse(data, offs)
print(f"resolved entries: {len(entries)} / {len(offs)}")

# data/297: raw bytes to EOF (complete tensor flushed, descriptor missing)
last = entries[-1]
assert last[0] == "archive/data/296", last[0]
ds297 = 0
for off in offs:
    nl, el = struct.unpack("<HH", data[off + 26 : off + 30])
    if data[off + 30 : off + 30 + nl].decode("utf-8", "replace") == "archive/data/297":
        ds297 = off + 30 + nl + el
        break
raw297 = data[ds297:]
print(f"data/297 raw bytes at EOF: {len(raw297)} (fp32 elems: {len(raw297)//4})")
entries.append(["archive/data/297", 0, len(raw297), ds297, len(data)])

recs = tiny_records()
print("version bytes:", recs.get("version", b"")[:8].hex(),
      "| serialization_id:", recs.get(".data/serialization_id", b"")[:16])

extra = {}
if "version" in recs:
    extra["archive/version"] = recs["version"]
if "serialization_id" in recs:
    extra["archive/.data/serialization_id"] = recs["serialization_id"]

rebuild(entries, extra, REC)
print("rebuilt:", REC, os.path.getsize(REC), "bytes, entries:", len(entries) + len(extra))

try:
    ck = torch.load(REC, map_location="cpu", weights_only=False)
    print("torch.load OK; keys:", list(ck.keys()))
except Exception as e:
    print("torch.load failed (first attempt):", type(e).__name__, e)
    # fallback: drop data/297 (last opt-state tensor), retry
    entries.pop()
    rebuild(entries, extra, REC)
    ck = torch.load(REC, map_location="cpu", weights_only=False)
    print("torch.load OK after dropping data/297; keys:", list(ck.keys()))

print("step:", ck["step"])
n = sum(t.numel() for t in ck["model"].values())
print("model params:", n, "expected 108160896")
assert n == 108160896 and ck["step"] == 1200

final = {
    "model": ck["model"],
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
assert n2 == 108160896
print("RECOVERY OK")
