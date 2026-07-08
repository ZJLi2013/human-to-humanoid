#!/usr/bin/env python3
"""Run the HaWoR front-end on one video and SAVE world_space_res.pth (no rendering).

demo.py computes pred_{trans,rot,hand_pose,betas,valid} but only feeds them to the
aitviewer/pytorch3d visualizer (the fragile ROCm part). For feature6 we only need the
world-space trajectory, so this driver reuses the same pipeline functions and dumps:

  <seq_folder>/world_space_res.pth   joblib [pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid]
                                     each (2,T,...), hand 0=left 1=right  (matches hawor_to_vitra.py)
  <seq_folder>/SLAM/hawor_slam_w_scale_<s>_<e>.npz   (written by hawor_slam)

  run (inside hawor-rocm:demo, cwd=/opt/HaWoR):
    python hawor_infer_egodex.py --video_path <mp4> --img_focal 736.6339
"""
import argparse
import os
import sys

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--img_focal", type=float, default=None, help="EgoDex GT focal (intrinsic[0,0]); omit to let HaWoR estimate")
    ap.add_argument("--input_type", default="file")
    ap.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    ap.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    ap.add_argument("--vis_mode", default="world")  # unused, kept for pipeline arg-compat
    ap.add_argument("--hawor_repo", default=".")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.hawor_repo))
    os.chdir(args.hawor_repo)

    import torch
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
    from scripts.scripts_test_video.hawor_slam import hawor_slam

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    print(f"[track] seq_folder={seq_folder} frames={start_idx}..{end_idx} n_img={len(imgfiles)}")

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    print(f"[motion] img_focal={img_focal}")

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    assert os.path.exists(slam_path), f"SLAM not produced: {slam_path}"
    print(f"[slam] {slam_path}")

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    def np_(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    out = [np_(pred_trans), np_(pred_rot), np_(pred_hand_pose), np_(pred_betas), np_(pred_valid)]
    dst = os.path.join(seq_folder, "world_space_res.pth")
    joblib.dump(out, dst)
    shapes = {n: tuple(a.shape) for n, a in
              zip(["trans", "rot", "hand_pose", "betas", "valid"], out)}
    nan = any(np.isnan(a).any() for a in out[:4])
    print(f"[save] {dst}  shapes={shapes}  L_valid={int(out[4][0].sum())} R_valid={int(out[4][1].sum())}  NaN={nan}")
    print(f"SEQ_FOLDER={seq_folder}")


if __name__ == "__main__":
    main()
