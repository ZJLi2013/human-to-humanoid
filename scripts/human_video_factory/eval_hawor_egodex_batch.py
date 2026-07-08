#!/usr/bin/env python3
"""Batch HaWoR-vs-EgoDex accuracy: iterate all runs/<task>__<idx>/world_space_res.pth, match the
EgoDex GT hdf5 (test/<task>/<idx>.hdf5), compute per-hand metrics, aggregate.

Reuses eval_hawor_vs_egodex.eval_hand. Headline = RTE (wrist, order-free). WA-MPJPE / tip-MPJPE
use per-clip Hungarian tip matching (order-free but mildly optimistic; disclosed). Also reports the
modal tip permutation across clips to lock the MANO-tip<->ARKit-finger order later.

  run (inside hawor-rocm:demo, cwd=/opt/HaWoR):
    python eval_hawor_egodex_batch.py --runs /data/overnight-tests/hawor/runs \
        --egodex /data/overnight-tests/egodex/test --hawor_repo /opt/HaWoR \
        --out_json /data/overnight-tests/hawor/eval_batch.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hawor_vs_egodex import eval_hand, hawor_world_joints  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/data/overnight-tests/hawor/runs")
    ap.add_argument("--egodex", default="/data/overnight-tests/egodex/test")
    ap.add_argument("--hawor_repo", default="/opt/HaWoR")
    ap.add_argument("--conf_thr", type=float, default=0.0)
    ap.add_argument("--out_json", default="/data/overnight-tests/hawor/eval_batch.json")
    args = ap.parse_args()

    seqs = sorted(glob.glob(os.path.join(args.runs, "*", "world_space_res.pth")))
    print(f"[batch-eval] {len(seqs)} seqs under {args.runs}")

    rows, perms = [], []
    for wp in seqs:
        seq = os.path.dirname(wp)
        cid = os.path.basename(seq)                       # <task>__<idx>
        task, _, idx = cid.rpartition("__")
        gt = os.path.join(args.egodex, task, f"{idx}.hdf5")
        if not os.path.exists(gt):
            print(f"  [skip] {cid}: no GT {gt}"); continue
        try:
            jw, valid = hawor_world_joints(args.hawor_repo, seq)
            for side, hidx in (("left", 0), ("right", 1)):
                r = eval_hand(side, hidx, jw, valid, gt, args)
                r["clip"] = cid
                rows.append(r)
                if "tip_perm" in r:
                    perms.append((side, tuple(r["tip_perm"][k] for k in sorted(r["tip_perm"]))))
        except Exception as e:
            print(f"  [err] {cid}: {e}")

    def agg(key, side=None):
        v = [r[key] for r in rows if key in r and (side is None or r["side"] == side)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        if not v:
            return None
        return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                "std": float(np.std(v)), "p90": float(np.percentile(v, 90))}

    summary = {"n_clips": len(seqs), "n_hand_evals": len(rows)}
    for key in ("rte_mm", "wa_mpjpe_mm", "tip_mpjpe_mm"):
        summary[key] = {"all": agg(key), "left": agg(key, "left"), "right": agg(key, "right")}
    for side in ("left", "right"):
        ps = [p for s, p in perms if s == side]
        if ps:
            summary[f"modal_tip_perm_{side}"] = {"perm": list(Counter(ps).most_common(1)[0][0]),
                                                 "count": Counter(ps).most_common(1)[0][1], "n": len(ps)}

    json.dump({"summary": summary, "rows": rows}, open(args.out_json, "w"), indent=1)
    print("\n=== AGGREGATE (mm) ===")
    for key in ("rte_mm", "wa_mpjpe_mm", "tip_mpjpe_mm"):
        a = summary[key]["all"]
        if a:
            print(f"  {key:14s} n={a['n']:3d}  mean={a['mean']:7.1f}  median={a['median']:7.1f}  "
                  f"std={a['std']:6.1f}  p90={a['p90']:7.1f}")
    for side in ("left", "right"):
        k = f"modal_tip_perm_{side}"
        if k in summary:
            print(f"  {k}: {summary[k]}")
    print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()
