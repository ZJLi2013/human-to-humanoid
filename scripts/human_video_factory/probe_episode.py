#!/usr/bin/env python3
"""Throwaway probe: dump the on-disk schema of a real VITRA-1M episode + index + stats,
so hawor_to_vitra.py can match it exactly. Run on banff against data_ssv2_10k."""
import glob
import json
import os
import sys

import numpy as np


def shp(x):
    if isinstance(x, np.ndarray):
        return f"ndarray{tuple(x.shape)} {x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]" + (f" e0={x[0]!r}"[:60] if x else "")
    if isinstance(x, dict):
        return "dict{" + ",".join(list(x.keys())[:12]) + "}"
    return f"{type(x).__name__}: {str(x)[:60]}"


def walk(d, prefix=""):
    for k, v in d.items():
        print(f"  {prefix}{k:26s} {shp(v)}")
        if isinstance(v, dict) and prefix == "":
            walk(v, prefix=f"{k}.")


root = sys.argv[1] if len(sys.argv) > 1 else "/data/overnight-tests/vitra/data_ssv2_10k"
print(f"=== root: {root} ===")
for p in sorted(glob.glob(os.path.join(root, "**"), recursive=True))[:40]:
    if os.path.isdir(p):
        print("[dir]", p)

# index
idx = glob.glob(os.path.join(root, "**", "episode_frame_index.npz"), recursive=True)
print("\n=== episode_frame_index.npz ===", idx[:1])
if idx:
    z = np.load(idx[0], allow_pickle=True)
    for k in z.files:
        a = z[k]
        print(f"  {k:22s} ndarray{tuple(a.shape)} {a.dtype}  e0={a.reshape(-1)[0] if a.size else None!r}")

# stats
st = glob.glob(os.path.join(root, "**", "*statistics*.json"), recursive=True)
print("\n=== statistics json ===", st[:1])
if st:
    s = json.load(open(st[0]))
    for k, v in s.items():
        n = len(v.get("mean", [])) if isinstance(v, dict) else "?"
        print(f"  {k:22s} keys={list(v.keys()) if isinstance(v,dict) else v}  len(mean)={n}")

# one episode
eps = glob.glob(os.path.join(root, "**", "episodic_annotations", "*.npy"), recursive=True)
print(f"\n=== episode .npy ({len(eps)} found) ===", eps[:1])
if eps:
    ep = np.load(eps[0], allow_pickle=True).item()
    print("top-level type:", type(ep).__name__)
    walk(ep)
