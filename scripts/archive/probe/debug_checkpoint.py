"""Debug tool: inspect truncated torch checkpoint internals (report to file)."""
import struct
import os
import zipfile
import zlib
import tempfile
import torch

p = r"F:\自研大模型\runs\a003_p_bot_s2\last.pt"
report = r"F:\自研大模型\runs\a003_p_bot_s2\_recovery_report.txt"
out = []

data = open(p, "rb").read()
fsize = len(data)

offs = []
i = 0
while True:
    i = data.find(b"PK\x03\x04", i)
    if i < 0:
        break
    offs.append(i)
    i += 4

out.append(f"file size: {fsize}")
out.append(f"local headers: {len(offs)}")

out.append("=== ALL ENTRY NAMES ===")
for idx, off in enumerate(offs):
    nl, el = struct.unpack("<HH", data[off + 26 : off + 30])
    out.append(f"  {idx}: {data[off+30:off+30+nl].decode('utf-8','replace')}")

out.append("=== canonical torch layout (tiny save, this env) ===")
tmp = os.path.join(tempfile.gettempdir(), "layout_test.pt")
torch.save({"a": torch.zeros(5)}, tmp)
z = zipfile.ZipFile(tmp)
out.append(str(z.namelist()))
# what does torch.load require? try stripping 'version' variants from tiny save
z2 = zipfile.ZipFile(tmp)
names = z2.namelist()
for cand in ("version", "archive/version"):
    if cand in names:
        out.append(f"tiny save HAS record: {cand}")

out.append("=== data/297 (last entry) tail debug ===")
off = offs[-1]
nl, el = struct.unpack("<HH", data[off + 26 : off + 30])
name = data[off + 30 : off + 30 + nl].decode("utf-8", "replace")
flags = struct.unpack("<H", data[off + 6 : off + 8])[0]
ds = off + 30 + nl + el
out.append(f"last header @ {off} name={name} flags={hex(flags)} ds={ds}")
out.append(f"bytes after header: {fsize - off}")
out.append(f"last 40 bytes hex: {data[-40:].hex()}")

# Try descriptor hypotheses at file end
for dl, lab in ((16, "16B(sig+crc+csize+usize)"), (12, "12B(crc+csize+usize)")):
    if dl == 16:
        d_sig, d_crc, d_csize, d_usize = struct.unpack("<IIII", data[fsize - dl : fsize])
    else:
        d_crc, d_csize, d_usize = struct.unpack("<III", data[fsize - dl : fsize])
    data_len_hyp = fsize - dl - ds
    crc_ok = (zlib.crc32(data[ds : fsize - dl]) & 0xFFFFFFFF) == d_crc
    out.append(
        f"  {lab}: csize={d_csize} usize={d_usize} crc={d_crc:08x} "
        f"| implied_data_len={data_len_hyp} | crc_match={crc_ok}"
    )

out.append("=== data/296 descriptor pattern ===")
off2 = offs[-2]
nl2, el2 = struct.unpack("<HH", data[off2 + 26 : off2 + 30])
name2 = data[off2 + 30 : off2 + 30 + nl2].decode("utf-8", "replace")
ds2 = off2 + 30 + nl2 + el2
out.append(f"  {name2} header @ {off2} ds={ds2} next_header@{offs[-1]}")
for dl in (16, 12):
    if dl == 16:
        d_sig, d_crc, d_csize, d_usize = struct.unpack("<IIII", data[offs[-1] - dl : offs[-1]])
        ok = d_sig == 0x08074B50 and d_csize == offs[-1] - dl - ds2 and (
            zlib.crc32(data[ds2 : offs[-1] - dl]) & 0xFFFFFFFF
        ) == d_crc
    else:
        d_crc, d_csize, d_usize = struct.unpack("<III", data[offs[-1] - dl : offs[-1]])
        ok = d_csize == offs[-1] - dl - ds2 and (
            zlib.crc32(data[ds2 : offs[-1] - dl]) & 0xFFFFFFFF
        ) == d_crc
    out.append(f"  dl={dl}: csize={d_csize} usize={d_usize} crc={d_crc:08x} OK={ok}")

open(report, "w", encoding="utf-8").write("\n".join(out))
print("report written to", report)
