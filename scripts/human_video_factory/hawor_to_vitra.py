#!/usr/bin/env python3
"""HaWoR world-space output -> VITRA-1M episode format (feature3 / P1 glue).

Schema verified against the real VITRA loader (vitra/datasets/{dataset,human_dataset}.py)
and a real ssv2 episode on banff. The loader reads ONLY these episode fields:
  extrinsics (T,4,4) w2c | intrinsics (3,3) | video_decode_frame (T,) | video_name |
  anno_type | text | text_rephrase | per-hand {beta(10), global_orient_worldspace(T,3,3),
  hand_pose(T,15,3,3), transl_worldspace(T,3), joints_worldspace(T,21,3), kept_frames(T,)}
It DERIVES camera-space itself (joints_manospace = R_mano^T @ (joints_world - transl_worldspace)),
so transl_worldspace is the MANO translation (= HaWoR pred_trans), NOT the wrist joint.
camspace/wrist/max_*_movement fields are crop byproducts and are omitted.

Input (HaWoR demo.py per-video seq_folder):
  world_space_res.pth  joblib list[5]: pred_trans/rot/hand_pose/betas/valid, (2,T,...)  hand 0=L 1=R
  SLAM/hawor_slam_w_scale_*.npz  traj(T,7 xyzw), scale, img_focal, img_center

Output layout the stock loader consumes (dataset_name must be 'ssv2'):
  <out>/Annotation/ssv2/episodic_annotations/<episode_id>.npy
  <out>/Annotation/ssv2/episode_frame_index.npz
  <out>/Annotation/statistics/ssv2_angle_statistics.json     (identity stats, smoke)
  <out>/Video/Somethingsomething-v2_root/<video_name>.webm   (only if load_images=True)

MANO forward (joints_worldspace) uses HaWoR run_mano_twohands (--hawor_repo).
Mapping notes: human-to-humanoid/docs/experiments/part3-exp.md.
"""
import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np


def axis_angle_to_matrix(aa):
    """(...,3) axis-angle -> (...,3,3) rotation matrix (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / (theta + 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], axis=-1).reshape(aa.shape[:-1] + (3, 3))
    th = theta[..., 0][..., None, None]
    I = np.broadcast_to(np.eye(3), aa.shape[:-1] + (3, 3))
    return I + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def quat_xyzw_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_slam_cam(npz_path):
    """HaWoR load_slam_cam reimpl -> R_w2c (T,3,3), t_w2c (T,3), focal, center(2)."""
    d = np.load(npz_path)
    traj = np.asarray(d["traj"], np.float64)                 # (T,7): t(3) + q_xyzw(4)
    scale = float(d["scale"])
    t_c2w = traj[:, :3] * scale
    R_c2w = quat_xyzw_to_matrix(traj[:, 3:])
    R_w2c = np.transpose(R_c2w, (0, 2, 1))
    t_w2c = -np.einsum("tij,tj->ti", R_w2c, t_c2w)
    return R_w2c, t_w2c, float(d["img_focal"]), np.asarray(d["img_center"], np.float64)


def mano_joints_world(hawor_repo, pred_trans, pred_rot, pred_hand_pose, pred_betas):
    """HaWoR run_mano_twohands -> world joints (2,T,21,3)."""
    sys.path.insert(0, hawor_repo)
    import torch
    from hawor.utils.process import run_mano_twohands
    cuda = torch.cuda.is_available()
    t = lambda x: (torch.as_tensor(np.asarray(x), dtype=torch.float32).cuda() if cuda
                   else torch.as_tensor(np.asarray(x), dtype=torch.float32))
    out = run_mano_twohands(t(pred_trans), t(pred_rot), t(pred_hand_pose), None, t(pred_betas), use_cuda=cuda)
    j = out["joints"]
    return np.asarray(j.detach().cpu() if hasattr(j, "detach") else j, np.float64)


def build_hand(h, pred_trans, pred_rot, pred_hand_pose, pred_betas, valid, joints):
    T = pred_rot.shape[1]
    v = valid[h] > 0
    beta = (pred_betas[h][v].mean(0) if v.any() else pred_betas[h].mean(0)).astype(np.float64)
    return {
        "beta": beta,
        "global_orient_worldspace": axis_angle_to_matrix(pred_rot[h]),                       # (T,3,3)
        "hand_pose": axis_angle_to_matrix(pred_hand_pose[h].reshape(T, 15, 3)),              # (T,15,3,3)
        "transl_worldspace": pred_trans[h].astype(np.float64),                               # (T,3) MANO transl
        "joints_worldspace": joints[h].astype(np.float64),                                   # (T,21,3)
        "kept_frames": valid[h].astype(np.int64),                                            # (T,)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor_out", required=True, help="HaWoR seq_folder (world_space_res.pth + SLAM/)")
    ap.add_argument("--hawor_repo", default=None, help="HaWoR repo root (MANO forward); omit if --joints_npz given")
    ap.add_argument("--joints_npz", default=None, help="precomputed world joints (2,T,21,3) as npz key 'joints'; bypasses MANO forward (offline validation)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--episode_id", default="somethingsomethingv2_demo_ep_000000")
    ap.add_argument("--video_name", default="video_0")
    ap.add_argument("--caption", default="manipulate object")
    args = ap.parse_args()

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        [np.asarray(x, np.float32) for x in joblib.load(os.path.join(args.hawor_out, "world_space_res.pth"))]
    T = pred_rot.shape[1]
    print(f"[load] world_space_res.pth T={T} trans{pred_trans.shape} rot{pred_rot.shape} pose{pred_hand_pose.shape}")

    slam = sorted(glob.glob(os.path.join(args.hawor_out, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not slam:
        sys.exit(f"[err] no SLAM npz under {args.hawor_out}/SLAM/")
    R_w2c, t_w2c, focal, center = load_slam_cam(slam[0])
    assert R_w2c.shape[0] == T, f"SLAM T={R_w2c.shape[0]} != {T}"
    print(f"[load] {os.path.basename(slam[0])} focal={focal:.1f} center={center.tolist()}")

    if args.joints_npz:
        joints = np.asarray(np.load(args.joints_npz)["joints"], np.float64)
        print(f"[joints] injected{joints.shape} (MANO forward bypassed)")
    else:
        if not args.hawor_repo:
            sys.exit("[err] need --hawor_repo (MANO forward) or --joints_npz")
        joints = mano_joints_world(args.hawor_repo, pred_trans, pred_rot, pred_hand_pose, pred_betas)
        print(f"[mano] joints{joints.shape}")

    extrinsics = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    extrinsics[:, :3, :3] = R_w2c
    extrinsics[:, :3, 3] = t_w2c
    K = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
    anno_type = "right" if pred_valid[1].sum() >= pred_valid[0].sum() else "left"

    episode = {
        "extrinsics": extrinsics,
        "intrinsics": K,
        "video_decode_frame": np.arange(T, dtype=np.int64),
        "video_name": args.video_name,
        "anno_type": anno_type,
        "text": {s: [(args.caption, (0, T))] for s in ("left", "right")},
        "text_rephrase": {s: [([args.caption], (0, T))] for s in ("left", "right")},
        "left": build_hand(0, pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, joints),
        "right": build_hand(1, pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, joints),
    }

    ann = os.path.join(args.out_dir, "Annotation", "ssv2", "episodic_annotations")
    stat = os.path.join(args.out_dir, "Annotation", "statistics")
    os.makedirs(ann, exist_ok=True)
    os.makedirs(stat, exist_ok=True)
    np.save(os.path.join(ann, f"{args.episode_id}.npy"), episode, allow_pickle=True)
    np.savez(os.path.join(args.out_dir, "Annotation", "ssv2", "episode_frame_index.npz"),
             index_frame_pair=np.array([[0, i] for i in range(T)], dtype=np.uint32),
             index_to_episode_id=np.array([args.episode_id]))
    stats = {}
    for s in ("left", "right"):
        stats[f"state_{s}"] = {"mean": [0.0] * 61, "std": [1.0] * 61}
        stats[f"action_{s}"] = {"mean": [0.0] * 51, "std": [1.0] * 51}
    json.dump(stats, open(os.path.join(stat, "ssv2_angle_statistics.json"), "w"), indent=1)

    rng = lambda x: f"[{np.nanmin(x):.3f},{np.nanmax(x):.3f}]"
    print(f"[save] {args.episode_id} anno_type={anno_type} L_valid={int(pred_valid[0].sum())}/{T} R_valid={int(pred_valid[1].sum())}/{T}")
    print(f"[save] transl_w L{rng(episode['left']['transl_worldspace'])} joints_w L{rng(episode['left']['joints_worldspace'])}")
    nan = any(np.isnan(episode[s][k]).any() for s in ("left", "right")
              for k in ("global_orient_worldspace", "hand_pose", "joints_worldspace", "transl_worldspace"))
    print(f"[save] wrote under {args.out_dir}  NaN_in_pose={nan}")


if __name__ == "__main__":
    main()
