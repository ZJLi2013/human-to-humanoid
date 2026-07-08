#!/usr/bin/env python3
"""Inspect one EgoDex hdf5: transforms joint names, intrinsic, N, attrs, confidences.
Informs the HaWoR<->EgoDex joint mapping in eval_hawor_vs_egodex.py.
  run (in a container with h5py):  python probe_egodex.py --root /data/overnight-tests/egodex/test
"""
import argparse, glob, os
import h5py, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/data/overnight-tests/egodex/test")
a = ap.parse_args()

h5s = sorted(glob.glob(os.path.join(a.root, "*", "*.hdf5")))
print(f"[scan] {len(h5s)} hdf5 files under {a.root}")
if not h5s:
    raise SystemExit("no hdf5")
p = h5s[0]
mp4 = p[:-5] + ".mp4"
print(f"[file] {p}  mp4_exists={os.path.exists(mp4)}")
with h5py.File(p, "r") as f:
    print("=== attrs ===")
    for k in f.attrs:
        v = f.attrs[k]
        print(f"  {k}: {str(v)[:120]}")
    print("=== camera ===")
    for k in f["camera"]:
        print(f"  camera/{k} shape={f['camera'][k].shape}")
    print("  intrinsic=\n", np.array(f["camera"]["intrinsic"]))
    tf = f["transforms"]
    names = list(tf.keys())
    print(f"=== transforms ({len(names)} joints), N={tf[names[0]].shape[0]} ===")
    for n in names:
        print("  ", n, tuple(tf[n].shape))
    if "confidences" in f:
        cf = f["confidences"]
        print(f"=== confidences ({len(list(cf.keys()))} joints) ===")
        print("  sample:", list(cf.keys())[:6])
    # sample wrist world positions (frame 0)
    for w in ("leftHand", "rightHand", "camera"):
        if w in tf:
            T0 = np.array(tf[w][0])
            print(f"  {w}[0] transl={T0[:3,3]}")
