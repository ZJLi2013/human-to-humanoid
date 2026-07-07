"""Build a viewable HaWoR-on-ROCm demo video from already-computed pipeline
outputs (no GL / no aitviewer). Overlays the ROCm-computed hand segmentation
masks (also the masked-DROID-SLAM input) and WiLoR detection boxes onto the
input frames.
"""
import os, glob
import numpy as np
import cv2

ROOT = "/work/HaWoR/example/video_0"
IMG_DIR = os.path.join(ROOT, "extracted_images")
TRK = os.path.join(ROOT, "tracks_0_121")
OUT_MP4 = os.path.join(ROOT, "hawor_rocm_overlay.mp4")
SAMPLE_DIR = os.path.join(ROOT, "overlay_samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

masks = np.load(os.path.join(TRK, "model_masks.npy"))          # (N,H,W) bool
tracks = np.load(os.path.join(TRK, "model_tracks.npy"), allow_pickle=True).item()

imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
N = min(len(imgs), masks.shape[0])
print("frames:", N, "mask shape:", masks.shape)

# per-frame box list: frame_idx -> list of (box, handedness, track_id)
by_frame = {}
for tid, entries in tracks.items():
    for e in entries:
        if not e.get("det", False):
            continue
        fr = int(e["frame"])
        box = np.asarray(e["det_box"]).reshape(-1)[:4]
        hand = int(np.asarray(e["det_handedness"]).reshape(-1)[0])  # 0=left,1=right
        by_frame.setdefault(fr, []).append((box, hand, int(tid)))

h0, w0 = cv2.imread(imgs[0]).shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(OUT_MP4, fourcc, 30.0, (w0, h0))

MASK_COLOR = np.array([0, 200, 255], dtype=np.uint8)   # amber (BGR) for hand region
L_COLOR = (80, 220, 80)     # left hand box - green
R_COLOR = (60, 60, 240)     # right hand box - red

for i in range(N):
    img = cv2.imread(imgs[i])
    if img is None:
        continue
    if (img.shape[0], img.shape[1]) != (w0 and h0, w0):
        pass
    # resize mask to image size if needed
    m = masks[i]
    if m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    # blend mask
    overlay = img.copy()
    overlay[m] = (0.45 * MASK_COLOR + 0.55 * overlay[m]).astype(np.uint8)
    img = overlay
    # boxes
    for box, hand, tid in by_frame.get(i, []):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = R_COLOR if hand == 1 else L_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = ("R" if hand == 1 else "L") + f"#{tid}"
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    # banner
    cv2.rectangle(img, (0, 0), (w0, 44), (0, 0, 0), -1)
    cv2.putText(img, f"HaWoR on AMD ROCm (MI300/gfx942)  frame {i:03d}/{N-1}"
                     "  |  hand mask + WiLoR detection",
                (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(img)
    if i in (0, N // 3, 2 * N // 3, N - 1):
        cv2.imwrite(os.path.join(SAMPLE_DIR, f"sample_{i:04d}.jpg"), img)

vw.release()
sz = os.path.getsize(OUT_MP4)
print("WROTE", OUT_MP4, sz, "bytes")
print("SAMPLES", sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg"))))
