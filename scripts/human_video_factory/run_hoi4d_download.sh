#!/usr/bin/env bash
# Fetch a MINIMAL HOI4D slice for feature8 L2 baseline (Tier 0/1, unified dataset).
#
# Official mirror is ONEDRIVE (hoi4d.github.io), NOT Google Drive. OneDrive anonymous share
# links are turned into direct downloads via the shares API:
#   https://api.onedrive.com/v1.0/shares/u!<base64url(share_url)>/root/content
# (Baidu mirror pan.baidu.com/s/1ZeX8SzU-gnJ-wpO9ZukulQ?pwd=7gfs exists but is unreachable
#  from this network; the remote GPU node CAN reach OneDrive.)
#
# We pull only what VITRA->retarget->sim needs and SKIP depth/pointcloud/seg:
#   RGB Video (feeds VITRA)   CAD Model (Tier 1 mesh)   Annotations (object 6D pose)
#   Camera Parameters         [optional] Hand Pose (GT hand)
# RGB has no per-category download -> download whole, keep ONLY one rigid category's
# C-subtree (ZY*/H*/C<CAT_CODE>/...) -> ~5-8GB on disk.
#
# HOI4D C* class mapping (index = C number), rigid ones marked *:
#   C1 ToyCar*  C2 Mug*  C3 Laptop  C4 StorageFurniture  C5 Bottle*  C6 Safe  C7 Bowl*
#   C8 Bucket*  C9 Scissors  C11 Pliers  C12 Kettle*  C13 Knife*  C14 TrashCan
#   C17 Lamp  C18 Stapler  C20 Chair
#
#   run:  bash run_hoi4d_download.sh                 # smoke: small components only
#         WITH_RGB=1 CAT_CODE=C5 bash run_hoi4d_download.sh   # + RGB, keep Bottle only
#   env:  HOI4D_DIR (default /data/zhengjli/hoi4d)  CAT_CODE (e.g. C5)  WITH_RGB (0/1)
#         WITH_HANDPOSE (0/1, default 0)
set -uo pipefail

HOI4D_DIR="${HOI4D_DIR:-/data/zhengjli/hoi4d}"
CAT_CODE="${CAT_CODE:-}"           # e.g. C5 ; empty => keep ALL categories (large)
WITH_RGB="${WITH_RGB:-0}"
WITH_HANDPOSE="${WITH_HANDPOSE:-0}"

# --- OneDrive anonymous share URLs (from hoi4d.github.io, 2026-07) ---
URL_RGB='https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIvAQAAAAAB4AcDOxj_uuh7alRaR9b7MQ?e=GyBNaD'
URL_CAD='https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIsAQAAAAAB5gBbXuK7eDzXQmzDS5W09g?e=DdkFTE'
URL_ANNO='https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIuAQAAAAABELgpzxDxCz58Qov76-wpvw?e=qhyY0h'
URL_CAM='https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIqAQAAAAABZBWQ3p5gxd-tu_3rOeSo_A?e=GjWeg8'
URL_HAND='https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIrAQAAAAABHyHqDhs5CpJAkWTAQoGWxQ?e=ALFMCD'

mkdir -p "$HOI4D_DIR" || { echo "cannot mkdir $HOI4D_DIR (use a user-writable path)"; exit 1; }
cd "$HOI4D_DIR"

onedrive_dl() {  # $1=share_url $2=out_file  (resumable, idempotent)
  local url="$1" out="$2"
  if [ -s "$out" ]; then echo "  skip $out ($(du -h "$out" | cut -f1))"; return 0; fi
  local b64
  b64=$(printf '%s' "$url" | base64 -w0 | tr '/+' '_-' | tr -d '=')
  echo "  download -> $out"
  curl -fL -m 7200 --retry 5 -C - \
    "https://api.onedrive.com/v1.0/shares/u!${b64}/root/content" -o "$out.part" \
    && mv -f "$out.part" "$out" \
    && echo "  ok $out ($(du -h "$out" | cut -f1))" \
    || { echo "  DL_FAIL $out"; return 1; }
}

echo "=== [1] small components (CAD + Annotations + Camera[, HandPose]) ==="
onedrive_dl "$URL_CAD"  cad_model.zip
onedrive_dl "$URL_ANNO" annotations.zip
onedrive_dl "$URL_CAM"  camera_params.zip
[ "$WITH_HANDPOSE" = "1" ] && onedrive_dl "$URL_HAND" hand_pose.zip

echo "=== [2] extract small components ==="
for z in cad_model annotations camera_params hand_pose; do
  [ -s "$z.zip" ] || continue
  echo "  unzip $z.zip"
  mkdir -p "$z"; unzip -o -q "$z.zip" -d "$z" || echo "  unzip_warn $z (maybe tar? inspecting)"
done
echo "  sizes:"; du -sh ./*.zip 2>/dev/null; ls -1

if [ "$WITH_RGB" = "1" ]; then
  echo "=== [3] RGB Video (large; ~40-60GB) ==="
  onedrive_dl "$URL_RGB" rgb_video.zip
  echo "=== [4] extract RGB, filter to CAT_CODE=${CAT_CODE:-<ALL>} ==="
  mkdir -p rgb
  if [ -n "$CAT_CODE" ]; then
    unzip -o -q rgb_video.zip -d rgb "*/${CAT_CODE}/*" \
      || { echo "  pattern extract failed -> full extract"; unzip -o -q rgb_video.zip -d rgb; }
    if [ -z "$(find rgb -name image.mp4 2>/dev/null | head -1)" ]; then
      echo "  pattern matched nothing -> full extract then prune"
      unzip -o -q rgb_video.zip -d rgb
      find rgb -type d -name 'C*' | grep -v "/${CAT_CODE}\$" | xargs -r rm -rf
    fi
  else
    unzip -o -q rgb_video.zip -d rgb
  fi
  echo "  rgb videos: $(find rgb -name image.mp4 2>/dev/null | wc -l)  size: $(du -sh rgb | cut -f1)"
  find rgb -name image.mp4 2>/dev/null | head -3
else
  echo "=== [3] RGB skipped (set WITH_RGB=1 CAT_CODE=C5 to pull + filter) ==="
fi

echo "=== HOI4D slice under $HOI4D_DIR  ($(date)) ==="
du -sh "$HOI4D_DIR"/* 2>/dev/null
