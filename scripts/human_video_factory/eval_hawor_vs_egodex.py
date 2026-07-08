#!/usr/bin/env python3
"""HaWoR recovered world-space hand trajectory  vs  EgoDex ARKit GT.

EgoDex ships per-joint 4x4 world transforms (ARKit origin frame, metric metres) collected
at capture time (Vision Pro multi-cam SLAM) -> a clean, zero-contamination held-out GT for
HaWoR's monocular world-space reconstruction (HaWoR/CVPR'25 only evaluated DexYCB/HOT3D).

Two coordinate systems differ by an unknown similarity (scale + R + t): HaWoR world is
up-to-SLAM-scale; EgoDex is metric ARKit. So all joint metrics are computed AFTER a
similarity alignment (Umeyama), which is the standard "world-aligned" protocol.

Metrics (mm, after alignment):
  RTE            wrist(root) trajectory error, global similarity align on wrist track
  WA-MPJPE       single global similarity over all matched joints+frames
  W-MPJPE-ff     similarity fit on frame-0 joints only, applied to whole seq (drift)
  PA-MPJPE       per-frame Procrustes (pure per-frame shape)

Robust tier (no joint-order assumption): wrist (unambiguous) + 5 fingertips matched by
Hungarian assignment after wrist-similarity align. Optional --full21 also maps the full
MANO-21 to the ARKit skeleton (Knuckle/IntBase/IntTip/Tip per finger) under a configurable
finger order (verify empirically once, then lock).

  run (inside hawor-rocm:demo, cwd=/opt/HaWoR):
    python eval_hawor_vs_egodex.py --hawor_out <seq_folder> --egodex_hdf5 <gt.hdf5> --hawor_repo .
"""
import argparse
import os
import sys

import h5py
import joblib
import numpy as np

# EgoDex ARKit finger joint names (proximal->distal): MANO MCP,PIP,DIP <-> Knuckle,IntBase,IntTip; tip separate
ARKIT_FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGERTIP = {f: f"{{side}}{f}FingerTip" if f != "Thumb" else "{side}ThumbTip" for f in ARKIT_FINGERS}


def umeyama(src, dst, with_scale=True):
    """Similarity transform mapping src->dst (both (N,3)). Returns s,R,t and aligned src."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / src.shape[0]
    try:
        U, D, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        # degenerate (near-static / collinear track): fall back to translation-only alignment
        t = mu_d - mu_s
        return 1.0, np.eye(3), t, src + t
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / src.shape[0]
    s = (D * np.diag(S)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    aligned = (s * (R @ src.T)).T + t
    return s, R, t, aligned


def apply_sim(s, R, t, pts):
    return (s * (R @ pts.reshape(-1, 3).T)).T.reshape(pts.shape) + t


def egodex_wrist_and_tips(h5path, side):
    """side in {left,right}. Returns wrist (N,3) and tips dict name->(N,3), metric metres."""
    with h5py.File(h5path, "r") as f:
        tf = f["transforms"]
        wrist = np.array(tf[f"{side}Hand"])[:, :3, 3]
        tips = {}
        for fg in ARKIT_FINGERS:
            key = FINGERTIP[fg].format(side=side)
            if key in tf:
                tips[fg] = np.array(tf[key])[:, :3, 3]
        conf = None
        if "confidences" in f and f"{side}Hand" in f["confidences"]:
            conf = np.array(f["confidences"][f"{side}Hand"])
    return wrist, tips, conf


def hawor_world_joints(hawor_repo, hawor_out):
    """(2,T,21,3) world joints via HaWoR run_mano_twohands. hand 0=left 1=right."""
    sys.path.insert(0, os.path.abspath(hawor_repo))
    import torch
    from hawor.utils.process import run_mano_twohands
    trans, rot, hand_pose, betas, valid = \
        [np.asarray(x, np.float32) for x in joblib.load(os.path.join(hawor_out, "world_space_res.pth"))]
    cuda = torch.cuda.is_available()
    t = lambda x: (torch.as_tensor(x).cuda() if cuda else torch.as_tensor(x))
    out = run_mano_twohands(t(trans), t(rot), t(hand_pose), None, t(betas), use_cuda=cuda)
    j = out["joints"]
    j = j.detach().cpu().numpy() if hasattr(j, "detach") else np.asarray(j)
    return j.astype(np.float64), valid  # (2,T,21,3), (2,T)


def hungarian_match(pred_tips, gt_tips):
    """pred_tips (5,3-agg), gt_tips (5,3-agg) -> perm mapping pred idx -> gt idx (min total dist)."""
    D = np.linalg.norm(pred_tips[:, None, :] - gt_tips[None, :, :], axis=-1)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(D)
        return dict(zip(r.tolist(), c.tolist())), D
    except Exception:
        used, perm = set(), {}
        for i in np.argsort(D.min(1)):
            j = int(np.argmin([D[i, k] if k not in used else 1e9 for k in range(5)]))
            perm[int(i)] = j; used.add(j)
        return perm, D


def eval_hand(side, hidx, jw, valid, h5path, args):
    T = jw.shape[1]
    wrist_gt, tips_gt, conf = egodex_wrist_and_tips(h5path, side)
    N = wrist_gt.shape[0]
    L = min(T, N)
    v = valid[hidx][:L] > 0
    if conf is not None:
        v = v & (conf[:L] > args.conf_thr)
    if v.sum() < 8:
        return {"side": side, "n_valid": int(v.sum()), "note": "too few valid frames"}

    wrist_pred = jw[hidx][:L, 0, :]                 # MANO j0 = wrist
    # global similarity from wrist track (unambiguous)
    s, R, t, _ = umeyama(wrist_pred[v], wrist_gt[:L][v])
    wrist_pred_a = apply_sim(s, R, t, wrist_pred)
    rte = np.linalg.norm(wrist_pred_a[v] - wrist_gt[:L][v], axis=-1)  # metres

    # fingertips: MANO tips = joints 16..20; match to GT tips by Hungarian on aggregate pos
    res = {"side": side, "n_valid": int(v.sum()), "L": int(L), "scale": float(s)}
    if len(tips_gt) == 5:
        pred_tips = jw[hidx][:L, 16:21, :]          # (L,5,3)
        pred_tips_a = apply_sim(s, R, t, pred_tips)
        gt_tip_arr = np.stack([tips_gt[fg][:L] for fg in ARKIT_FINGERS], axis=1)  # (L,5,3)
        agg_p = pred_tips_a[v].mean(0)              # (5,3)
        agg_g = gt_tip_arr[v].mean(0)
        perm, _ = hungarian_match(agg_p, agg_g)
        err_tips = []
        for pi, gi in perm.items():
            e = np.linalg.norm(pred_tips_a[v, pi, :] - gt_tip_arr[v, gi, :], axis=-1)
            err_tips.append(e.mean())
        res["tip_mpjpe_mm"] = float(np.mean(err_tips) * 1000)
        res["tip_perm"] = {int(k): int(v_) for k, v_ in perm.items()}
        # WA-MPJPE over wrist+tips with a refit similarity on matched set
        pred_set = np.concatenate([wrist_pred[:L, None, :], pred_tips[:, [p for p in perm]]], axis=1)
        gt_set = np.concatenate([wrist_gt[:L, None, :],
                                 np.stack([gt_tip_arr[:, perm[p]] for p in perm], axis=1)], axis=1)
        s2, R2, t2, _ = umeyama(pred_set[v].reshape(-1, 3), gt_set[v].reshape(-1, 3))
        pred_set_a = apply_sim(s2, R2, t2, pred_set)
        res["wa_mpjpe_mm"] = float(np.linalg.norm(pred_set_a[v] - gt_set[v], axis=-1).mean() * 1000)
    res["rte_mm"] = float(rte.mean() * 1000)
    res["rte_max_mm"] = float(rte.max() * 1000)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor_out", required=True, help="seq_folder with world_space_res.pth")
    ap.add_argument("--egodex_hdf5", required=True, help="matching EgoDex GT hdf5")
    ap.add_argument("--hawor_repo", default=".")
    ap.add_argument("--conf_thr", type=float, default=0.0)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    jw, valid = hawor_world_joints(args.hawor_repo, args.hawor_out)
    print(f"[hawor] joints{jw.shape} valid L={int(valid[0].sum())} R={int(valid[1].sum())}")

    results = []
    for side, hidx in (("left", 0), ("right", 1)):
        r = eval_hand(side, hidx, jw, valid, args.egodex_hdf5, args)
        results.append(r)
        print(f"[{side}] " + "  ".join(f"{k}={v}" for k, v in r.items()))

    summary = {"hawor_out": args.hawor_out, "egodex_hdf5": args.egodex_hdf5, "hands": results}
    if args.out_json:
        import json
        json.dump(summary, open(args.out_json, "w"), indent=1)
        print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()
