#!/usr/bin/env bash
# Fetch a MINIMAL HOI4D slice for feature8 L2 baseline (Tier 0/1, unified dataset).
#
# HOI4D is distributed as SEPARATE component files on Google Drive (see
# github.com/leolyliu/HOI4D-Instructions). We pull only what the VITRA->retarget->sim
# pipeline needs and SKIP the big depth/pointcloud/seg components:
#   RGB Video  (feeds VITRA)          CAD Model  (Tier 1 object mesh)
#   Annotations(object 6D pose)       Camera Parameters (place object in cam frame)
#   [optional] Hand Pose (GT hand, for comparison only)
#
# RGB has NO official per-category download (one monolithic archive ~40-60GB); we download it
# whole, then keep ONLY one rigid category's C-subtree (ZY*/H*/C<CAT_CODE>/...) and delete the
# rest -> ~5-8GB on disk. Depth/pointcloud/seg are never downloaded.
#
#   run:  bash run_hoi4d_download.sh
#   env:  HOI4D_DIR (default /data/overnight-tests/hoi4d)
#         CAT_CODE  (rigid category folder index, e.g. C5; check definitions/task_definitions.csv
#                    in the annotations pack — map Bottle/Mug/ToyCar -> C-index BEFORE filtering)
#         WITH_HANDPOSE (0/1, default 0)
#
# NOTE: verify the Google Drive file IDs below against the HOI4D-Instructions README before a
# long pull — the project occasionally rotates links. IDs current as of 2026-07.
set -uo pipefail

HOI4D_DIR="${HOI4D_DIR:-/data/overnight-tests/hoi4d}"
CAT_CODE="${CAT_CODE:-}"           # e.g. C5 ; empty => keep ALL categories (large)
WITH_HANDPOSE="${WITH_HANDPOSE:-0}"
IMAGE="${IMAGE:-python:3.11-slim}"

# --- Google Drive file IDs (VERIFY against leolyliu/HOI4D-Instructions) ---
ID_RGB="1pL9V_x5tWk1QFjT7Dhiij1W-AXARyhPP"           # RGB Video (monolithic ~40-60GB)
ID_CAD="1evpTI51UpDg_a1kAGXQdkfSd0DlI1_Gl"           # CAD Model
ID_ANNO="1rQVoYsAilwmq66ilGj6d2Q2fUF7osrpn"          # Annotations (object pose + task_definitions.csv)
ID_CAM="__VERIFY_CAMERA_PARAM_ID__"                  # Camera Parameters (grab exact ID from repo)
ID_HAND="1tjNUKaM23QmkG_toLXjUVlVSJJAlcHBN"          # Hand Pose (optional)

U=$(id -u); G=$(id -g)
if [ ! -d "$HOI4D_DIR" ]; then
  echo "[setup] minting self-owned $HOI4D_DIR via alpine container"
  docker run --rm -v /data:/data alpine sh -c "mkdir -p '$HOI4D_DIR' && chown -R $U:$G '$HOI4D_DIR'"
fi
mkdir -p "$HOI4D_DIR"

echo "=== download components (gdown, resumable) -> $HOI4D_DIR ==="
docker run --rm -i -v /data:/data \
  -e HOI4D_DIR="$HOI4D_DIR" -e WITH_HANDPOSE="$WITH_HANDPOSE" \
  -e ID_RGB="$ID_RGB" -e ID_CAD="$ID_CAD" -e ID_ANNO="$ID_ANNO" -e ID_CAM="$ID_CAM" -e ID_HAND="$ID_HAND" \
  "$IMAGE" bash -s <<'INNER'
set -uo pipefail
pip install -q gdown >/dev/null 2>&1 || pip install gdown
cd "$HOI4D_DIR"
dl() {  # $1=id $2=out  (skip if present & non-empty)
  [ "$1" = "__VERIFY_CAMERA_PARAM_ID__" ] && { echo "  SKIP $2 (set ID_CAM first)"; return 0; }
  if [ -s "$2" ]; then echo "  skip $2 ($(du -h "$2"|cut -f1))"; return 0; fi
  echo "  gdown $2"; gdown -q "$1" -O "$2" || { echo "  DL_FAIL $2"; return 1; }
}
dl "$ID_ANNO" annotations.zip
dl "$ID_CAD"  cad_model.zip
dl "$ID_CAM"  camera_params.zip
[ "$WITH_HANDPOSE" = "1" ] && dl "$ID_HAND" hand_pose.zip
dl "$ID_RGB"  rgb_video.zip           # big; do last so small parts land first
echo "=== extract small components ==="
for z in annotations cad_model camera_params hand_pose; do
  [ -s "$z.zip" ] && { echo "unzip $z"; unzip -o -q "$z.zip" -d "$z" || echo "  unzip_warn $z"; }
done
echo "=== sizes ==="; du -sh ./*.zip 2>/dev/null; ls -1
INNER

echo "=== extract + filter RGB to one category (CAT_CODE=${CAT_CODE:-<ALL>}) ==="
if [ -s "$HOI4D_DIR/rgb_video.zip" ]; then
  docker run --rm -i -v /data:/data -e HOI4D_DIR="$HOI4D_DIR" -e CAT_CODE="$CAT_CODE" "$IMAGE" bash -s <<'INNER'
set -uo pipefail
cd "$HOI4D_DIR"
mkdir -p rgb
if [ -n "$CAT_CODE" ]; then
  # HOI4D path: ZY*/H*/C<code>/N*/S*/s*/T*/align_rgb/image.mp4  -> extract only matching C-folder
  echo "keep only /$CAT_CODE/ ; deleting others after extract"
  unzip -o -q rgb_video.zip -d rgb "*/${CAT_CODE}/*" || echo "  (pattern extract warn; falling back to full)"
  [ -z "$(find rgb -name image.mp4 2>/dev/null | head -1)" ] && { echo "  pattern empty -> full extract"; unzip -o -q rgb_video.zip -d rgb; }
else
  unzip -o -q rgb_video.zip -d rgb
fi
echo "rgb videos: $(find rgb -name image.mp4 2>/dev/null | wc -l)  size: $(du -sh rgb|cut -f1)"
find rgb -name image.mp4 2>/dev/null | head -3
INNER
else
  echo "[skip] rgb_video.zip not present (verify ID_RGB)"
fi

echo "=== HOI4D minimal slice ready under $HOI4D_DIR  ($(date)) ==="
du -sh "$HOI4D_DIR"/* 2>/dev/null
echo "NEXT: map Bottle/Mug/ToyCar -> C-index via annotations/definitions/task_definitions.csv, set CAT_CODE, re-run to filter."
