#!/usr/bin/env python3
"""Offline round-trip validation of hawor_to_vitra.py (feature3 / P1), no HaWoR/MANO needed.

real VITRA episode --reverse--> world_space_res.pth + SLAM npz + joints.npz
   --hawor_to_vitra.py--> new episode --> reload via stock FrameDataset --> diff vs original.

Exercises: matrix<->axis-angle, transl passthrough, extrinsics<->SLAM(traj) reconstruction,
on-disk layout/dtypes, and loadability through the real loader. MANO-forward joints are
injected (that path is HaWoR-native and runs on the real node)."""
import argparse
import glob
import os
import subprocess
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def mat_to_axis_angle(R):
    R = np.asarray(R, np.float64)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    ax = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    ax = np.where(s < 1e-8, np.array([1.0, 0.0, 0.0]), ax / (s + 1e-12))
    return ax * ang[..., None]


def mat_to_quat_xyzw(R):
    R = np.asarray(R, np.float64)
    w = np.sqrt(np.clip(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], 0, None)) / 2
    w = np.maximum(w, 1e-8)
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
    return np.stack([x, y, z, w], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitra_repo", required=True)
    ap.add_argument("--dataset_folder", required=True)
    ap.add_argument("--work", default="/tmp/hvf_roundtrip")
    args = ap.parse_args()

    src = sorted(glob.glob(os.path.join(args.dataset_folder, "Annotation", "ssv2", "episodic_annotations", "*.npy")))
    if not src:
        sys.exit("[err] no source episode")
    ep = np.load(src[0], allow_pickle=True).item()
    T = len(ep["extrinsics"])
    eid, vname = os.path.splitext(os.path.basename(src[0]))[0], ep["video_name"]
    print(f"[src] {eid} T={T} anno_type={ep['anno_type']}")

    # reverse -> HaWoR world_space_res.pth (hand 0=left,1=right)
    def stack(fn):
        return np.stack([fn(ep["left"]), fn(ep["right"])], 0)
    pred_trans = stack(lambda s: np.asarray(s["transl_worldspace"], np.float32))
    pred_rot = stack(lambda s: mat_to_axis_angle(s["global_orient_worldspace"]).astype(np.float32))
    pred_pose = stack(lambda s: mat_to_axis_angle(s["hand_pose"]).reshape(T, 45).astype(np.float32))
    pred_beta = stack(lambda s: np.tile(np.asarray(s["beta"], np.float32), (T, 1)))
    pred_valid = stack(lambda s: np.asarray(s["kept_frames"], np.float32))
    joints = stack(lambda s: np.asarray(s["joints_worldspace"], np.float64))

    ho = os.path.join(args.work, "hawor_out")
    os.makedirs(os.path.join(ho, "SLAM"), exist_ok=True)
    joblib.dump([pred_trans, pred_rot, pred_pose, pred_beta, pred_valid], os.path.join(ho, "world_space_res.pth"))
    np.savez(os.path.join(args.work, "joints.npz"), joints=joints)

    # reverse extrinsics(w2c) -> SLAM traj (c2w, xyzw), scale=1
    E = np.asarray(ep["extrinsics"], np.float64)
    R_w2c, t_w2c = E[:, :3, :3], E[:, :3, 3]
    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    t_c2w = -np.einsum("tij,tj->ti", R_c2w, t_w2c)
    traj = np.concatenate([t_c2w, mat_to_quat_xyzw(R_c2w)], -1)
    K = np.asarray(ep["intrinsics"], np.float64)
    np.savez(os.path.join(ho, "SLAM", f"hawor_slam_w_scale_0_{T}.npz"),
             traj=traj, scale=np.float64(1.0), img_focal=np.float64(K[0, 0]),
             img_center=np.array([K[0, 2], K[1, 2]], np.float64),
             tstamp=np.arange(T), disps=np.zeros((T, 1, 1)))

    # run converter
    out = os.path.join(args.work, "out")
    r = subprocess.run([sys.executable, os.path.join(HERE, "hawor_to_vitra.py"),
                        "--hawor_out", ho, "--joints_npz", os.path.join(args.work, "joints.npz"),
                        "--out_dir", out, "--episode_id", eid, "--video_name", str(vname)],
                       capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip()[-800:] if r.stderr else "")
    if r.returncode:
        sys.exit("[err] converter failed")

    # diff produced vs original
    prod = np.load(os.path.join(out, "Annotation", "ssv2", "episodic_annotations", f"{eid}.npy"), allow_pickle=True).item()
    print("\n[diff] produced vs original (max abs err):")
    ok = True
    for hand in ("left", "right"):
        for f in ("global_orient_worldspace", "hand_pose", "transl_worldspace", "joints_worldspace"):
            e = float(np.max(np.abs(np.asarray(prod[hand][f], np.float64) - np.asarray(ep[hand][f], np.float64))))
            ok &= e < 1e-3
            print(f"  {hand}.{f:26s} {e:.2e}")
    eE = float(np.max(np.abs(np.asarray(prod["extrinsics"], np.float64) - E)))
    print(f"  extrinsics                      {eE:.2e}"); ok &= eE < 1e-3

    # reload produced through stock loader
    sys.path.insert(0, args.vitra_repo)
    from vitra.datasets.dataset import FrameDataset
    ds = FrameDataset(dataset_folder=out, dataset_name="ssv2", processor=None,
                      action_past_window_size=0, action_future_window_size=15,
                      action_type="angle", use_rel=False, normalization=True,
                      augmentation=False, load_images=False, data_type="human")
    s = ds[0]
    print(f"\n[reload] len={len(ds)} sample: action_list{tuple(s['action_list'].shape)} current_state{tuple(s['current_state'].shape)}")
    print(f"\n[RESULT] round-trip {'PASS' if ok else 'FAIL'}: converter output matches source + loads through FrameDataset.")


if __name__ == "__main__":
    main()
