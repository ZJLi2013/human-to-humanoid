"""Visualize HaWoR's CORE output on ROCm: the world-space, continuous 3D hand
motion trajectory. Bypasses the hung aitviewer/GL renderer by:
  1. loading world_space_res.pth (world-frame MANO params from demo.py),
  2. running MANO forward on the ROCm GPU (run_mano_twohands) -> world verts/joints,
  3. drawing a rotating 3D hand-mesh turntable + wrist trajectory with matplotlib
     (Agg backend, pure software, no OpenGL).
Run from the HaWoR repo root.
"""
import os, sys, glob, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch, joblib

REPO = "/work/HaWoR"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "scripts_test_video"))
from hawor.utils.process import run_mano_twohands, get_mano_faces

R = os.path.join(REPO, "example", "video_0")
OUT_DIR = os.path.join(R, "world_traj")
FRAME_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(
    os.path.join(R, "world_space_res.pth"))
# to tensors (2, T, ...)
pt = torch.as_tensor(np.asarray(pred_trans)).float()
pr = torch.as_tensor(np.asarray(pred_rot)).float()
ph = torch.as_tensor(np.asarray(pred_hand_pose)).float()
pb = torch.as_tensor(np.asarray(pred_betas)).float()
valid = np.asarray(pred_valid).astype(bool)
T = pt.shape[1]
print("frames:", T, "valid L/R:", valid[0].sum(), valid[1].sum())

with torch.no_grad():
    out = run_mano_twohands(pt, pr, ph, None, pb, use_cuda=True)
verts = out["vertices"].detach().cpu().numpy()   # (2, T, 778, 3) world
joints = out["joints"].detach().cpu().numpy()     # (2, T, J, 3) world
faces = np.asarray(get_mano_faces())              # (F, 3)
print("verts", verts.shape, "joints", joints.shape, "faces", faces.shape)

wrist = joints[:, :, 0, :]                         # (2, T, 3)

# axis bounds from the wrist TRAJECTORY (robust percentiles so a stray infiller
# frame can't blow up the view), padded for hand size -> hands stay visible.
wpts = np.concatenate([wrist[h][valid[h]] for h in range(2)], 0)   # (M,3)
p5, p95 = np.percentile(wpts, 5, axis=0), np.percentile(wpts, 95, axis=0)
c = (p5 + p95) / 2
r = max((p95 - p5).max() / 2, 0.15) + 0.18         # pad for the ~0.15 hand
lims = np.stack([c - r, c + r], 1)                 # (3,2)

HAND_COLOR = {0: (0.20, 0.70, 0.30), 1: (0.85, 0.25, 0.25)}  # L green, R red
HAND_NAME = {0: "left", 1: "right"}


def draw_frame(t, azim, lims_f, trail_start, out_dir, title_sfx):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.90], projection="3d")
    for h in range(2):
        if not valid[h, t]:
            continue
        V = verts[h, t]
        mesh = Poly3DCollection(V[faces], alpha=0.55,
                                facecolor=HAND_COLOR[h], edgecolor="none")
        ax.add_collection3d(mesh)
        # wrist trajectory trail (trail_start..t; full history for the fixed view,
        # a short recent window for the follow-cam so it stays inside the tight box)
        ts = trail_start if trail_start is not None else 0
        tr = wrist[h, ts:t + 1]
        tv = valid[h, ts:t + 1]
        tr = tr[tv]
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=HAND_COLOR[h], lw=2.0)
        ax.scatter(wrist[h, t, 0], wrist[h, t, 1], wrist[h, t, 2],
                   color=HAND_COLOR[h], s=25)
    ax.set_xlim(lims_f[0]); ax.set_ylim(lims_f[1]); ax.set_zlim(lims_f[2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=15, azim=azim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.suptitle(f"HaWoR world-space 3D hand motion on AMD ROCm (MI300/gfx942){title_sfx}\n"
                 f"frame {t:03d}/{T-1}  |  green=left  red=right  |  trail=wrist trajectory",
                 fontsize=10, y=0.99)
    fp = os.path.join(out_dir, f"f_{t:04d}.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


def render_pass(out_dir, sample_prefix, follow):
    """follow=False: fixed world box + full trail. follow=True: camera centered on
    the current hand centroid (hands zoomed in) + short recent trail."""
    os.makedirs(out_dir, exist_ok=True)
    R_FOLLOW = 0.22          # tight radius -> hands fill the frame
    TRAIL_WIN = 25           # recent-frame trail window for follow-cam
    sample_idx = {0, T // 3, 2 * T // 3, T - 1}
    import shutil
    for t in range(T):
        azim = -70 + t * (140.0 / max(1, T - 1))   # slow sweep -70 -> +70
        if follow:
            cen = np.mean([wrist[h, t] for h in range(2) if valid[h, t]], axis=0)
            lims_f = np.stack([cen - R_FOLLOW, cen + R_FOLLOW], 1)
            trail_start = max(0, t - TRAIL_WIN)
            title_sfx = "  [follow-cam]"
        else:
            lims_f = lims
            trail_start = None
            title_sfx = ""
        fp = draw_frame(t, azim, lims_f, trail_start, out_dir, title_sfx)
        if t in sample_idx:
            shutil.copy(fp, os.path.join(OUT_DIR, f"{sample_prefix}_{t:04d}.png"))
        if t % 20 == 0:
            print(f"[{sample_prefix}] rendered frame", t)


render_pass(FRAME_DIR, "sample", follow=False)
FOLLOW_DIR = os.path.join(OUT_DIR, "frames_follow")
render_pass(FOLLOW_DIR, "followcam", follow=True)

# ---- ego-view: follow the egocentric (SLAM) camera so the 3D aligns with the 2D
# overlay (same viewing direction -> left/right no longer mirrored). Transform the
# world-space mesh into camera coords per frame, then remap to mpl axes
# (x = cam-right, y = cam-depth, z = up) and keep a slight tilt for 3D depth. ----
from lib.eval_utils.custom_utils import load_slam_cam
_slam = sorted(glob.glob(os.path.join(R, "SLAM", "hawor_slam_w_scale_*.npz")))[0]
Rw2c, tw2c, Rc2w, tc2w = [np.asarray(a) for a in load_slam_cam(_slam)]


def _remap(Vc):   # cam (x right, y down, z fwd) -> mpl (x right, y depth, z up)
    return np.stack([Vc[..., 0], Vc[..., 2], -Vc[..., 1]], -1)


vego = np.zeros_like(verts)   # (2,T,778,3) in mpl-ego coords
wego = np.zeros_like(wrist)   # (2,T,3)
for h in range(2):
    for t in range(T):
        if not valid[h, t]:
            continue
        Vc = np.einsum("ij,nj->ni", Rw2c[t], verts[h, t]) + tw2c[t]
        vego[h, t] = _remap(Vc)
        wego[h, t] = _remap((Rw2c[t] @ wrist[h, t]) + tw2c[t])

EGO_DIR = os.path.join(OUT_DIR, "frames_ego")
os.makedirs(EGO_DIR, exist_ok=True)
R_EGO = 0.18
for t in range(T):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.90], projection="3d")
    cen = np.mean([wego[h, t] for h in range(2) if valid[h, t]], axis=0)
    ts = max(0, t - 25)
    for h in range(2):
        if not valid[h, t]:
            continue
        ax.add_collection3d(Poly3DCollection(vego[h, t][faces], alpha=0.55,
                            facecolor=HAND_COLOR[h], edgecolor="none"))
        tr = wego[h, ts:t + 1][valid[h, ts:t + 1]]
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=HAND_COLOR[h], lw=2.0)
    ax.set_xlim(cen[0] - R_EGO, cen[0] + R_EGO)
    ax.set_ylim(cen[1] - R_EGO, cen[1] + R_EGO)
    ax.set_zlim(cen[2] - R_EGO, cen[2] + R_EGO)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=8, azim=-90)   # look into scene from behind the head camera
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.suptitle("HaWoR world-space 3D hand motion on AMD ROCm (MI300/gfx942)  [ego-view]\n"
                 f"frame {t:03d}/{T-1}  |  view follows the egocentric camera  |  green=left  red=right",
                 fontsize=10, y=0.99)
    fp = os.path.join(EGO_DIR, f"f_{t:04d}.png")
    fig.savefig(fp); plt.close(fig)
    if t in {0, T // 3, 2 * T // 3, T - 1}:
        import shutil
        shutil.copy(fp, os.path.join(OUT_DIR, f"egoview_{t:04d}.png"))
    if t % 20 == 0:
        print("[egoview] rendered frame", t)

# static full-clip trajectory figure
fig = plt.figure(figsize=(8, 6), dpi=110)
ax = fig.add_subplot(111, projection="3d")
for h in range(2):
    tr = wrist[h][valid[h]]
    if len(tr) > 1:
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=HAND_COLOR[h], lw=2,
                label=f"{HAND_NAME[h]} wrist")
        ax.scatter(*tr[0], color=HAND_COLOR[h], s=40, marker="o")
        ax.scatter(*tr[-1], color=HAND_COLOR[h], s=60, marker="^")
ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass
ax.view_init(elev=18, azim=-60)
ax.legend()
ax.set_title("HaWoR world-space wrist trajectories (full clip) on AMD ROCm")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "wrist_trajectory_3d.png"))
plt.close(fig)

# encode mp4s (ffmpeg): fixed-box world view + follow-cam
def encode(frame_dir, mp4):
    cmd = ["ffmpeg", "-y", "-framerate", "20", "-i",
           os.path.join(frame_dir, "f_%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=960:540", mp4]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    ok = os.path.exists(mp4)
    print("MP4:", mp4, os.path.getsize(mp4) if ok else "MISSING", rc.stderr[-200:] if not ok else "")

encode(FRAME_DIR, os.path.join(R, "hawor_rocm_world_hand_motion.mp4"))
encode(FOLLOW_DIR, os.path.join(R, "hawor_rocm_world_hand_motion_followcam.mp4"))
encode(EGO_DIR, os.path.join(R, "hawor_rocm_world_hand_motion_egoview.mp4"))
print("SAMPLES:", sorted(glob.glob(os.path.join(OUT_DIR, "sample_*.png"))))
print("FOLLOW_SAMPLES:", sorted(glob.glob(os.path.join(OUT_DIR, "followcam_*.png"))))
print("EGO_SAMPLES:", sorted(glob.glob(os.path.join(OUT_DIR, "egoview_*.png"))))
print("TRAJ_PNG:", os.path.join(OUT_DIR, "wrist_trajectory_3d.png"))
