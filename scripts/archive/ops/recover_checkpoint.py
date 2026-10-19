"""Recover a truncated torch checkpoint (missing zip central directory).

Usage:
    python scripts/recover_checkpoint.py <ckpt.pt> [--out recovered.pt]

Scans the raw file for zip local headers, resolves entry boundaries via
data descriptors (torch's C++ writer streams with flag bit 3), verifies
CRCs, rebuilds a valid zip, and writes it out. Prints a per-entry report.
"""
import struct
import sys
import os
import io
import zipfile
import zlib
import argparse


def parse_entries(data: bytes):
    fsize = len(data)
    offs = []
    i = 0
    while True:
        i = data.find(b"PK\x03\x04", i)
        if i < 0:
            break
        offs.append(i)
        i += 4
    entries = []
    unresolved = []
    for idx, off in enumerate(offs):
        sig, ver, flags, method, mt, md, crc, csize, usize, nl, el = struct.unpack(
            "<IHHHHHIIIHH", data[off : off + 30]
        )
        name = data[off + 30 : off + 30 + nl].decode("utf-8", "replace")
        ds = off + 30 + nl + el
        end = None
        if flags & 0x08:  # streaming: sizes live in a trailing descriptor
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
                        end = nxt - dl
                        crc, usize, csize = d_crc, d_usize, d_csize
                        break
                if end is not None:
                    break
            if end is None:
                unresolved.append((idx, name, off, flags))
                continue
        else:
            end = ds + csize
        if end > fsize:
            unresolved.append((idx, name, off, "OOB"))
            continue
        entries.append((name, crc, usize, ds, end))
    return entries, unresolved, offs


def rebuild(entries, out_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, crc, usize, ds, end in entries:
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, data[ds:end])
    open(out_path, "wb").write(buf.getvalue())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    path = args.ckpt
    data = open(path, "rb").read()
    print("file size:", len(data))
    print("first 4 bytes:", data[:4])

    entries, unresolved, offs = parse_entries(data)
    print("local headers:", len(offs), "| resolved entries:", len(entries),
          "| unresolved:", len(unresolved))
    for u in unresolved:
        print("  UNRESOLVED:", u)
    print("first 8:", [e[0] for e in entries[:8]])
    print("last 8:", [e[0] for e in entries[-8:]])

    out = args.out or path.replace(".pt", "_recovered.pt")
    rebuild(entries, out)
    print("rebuilt:", out, os.path.getsize(out), "bytes, entries:", len(entries))
