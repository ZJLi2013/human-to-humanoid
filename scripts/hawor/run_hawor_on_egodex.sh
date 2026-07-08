#!/usr/bin/env bash
# Run the HaWoR front-end on EgoDex mp4 clips (ROCm, baked hawor-rocm:demo image) and save
# world_space_res.pth per clip. Manages one persistent container `hawor_run`; weights are
# hf-cached on /data (survive container recreation) and copied into /opt/HaWoR on start.
#
#   bash run_hawor_on_egodex.sh --setup                              # container + weights only
#   bash run_hawor_on_egodex.sh --video /data/.../test/task/0.mp4 --focal 736.6339 --id task__0
#
# Output: /data/overnight-tests/hawor/runs/<id>/{world_space_res.pth, SLAM/, extracted_images/, tracks_*/}
set -uo pipefail

IMAGE="${IMAGE:-hawor-rocm:demo}"
CNAME="${CNAME:-hawor_run}"
WORK="${WORK:-/data/overnight-tests/hawor}"
GPU="${HIP_VISIBLE_DEVICES:-1}"
RUNS="$WORK/runs"; HF="$WORK/hf"

U=$(id -u); G=$(id -g)
mint() { docker run --rm -v /data:/data alpine sh -c "mkdir -p '$1' && chown -R $U:$G '$1'" 2>/dev/null; }
mint "$WORK"; mint "$RUNS"; mint "$HF"

ensure_container() {
  if ! docker ps --format '{{.Names}}' | grep -qx "$CNAME"; then
    docker rm -f "$CNAME" 2>/dev/null || true
    echo "[container] starting $CNAME (GPU $GPU)"
    docker run -dit --name "$CNAME" \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 16g \
      -e HIP_VISIBLE_DEVICES="$GPU" -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
      -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -e HF_ENDPOINT=https://hf-mirror.com \
      -v /data:/data "$IMAGE" sleep infinity >/dev/null
  fi
}

setup_weights() {
  docker exec "$CNAME" bash -lc '
    set -e; cd /opt/HaWoR
    HF=/data/overnight-tests/hawor/hf
    if [ ! -f "$HF/.hawor_done" ]; then
      echo "[weights] hf download ThunderVVV/HaWoR"
      hf download ThunderVVV/HaWoR --repo-type model --local-dir "$HF/hawor" >/data/overnight-tests/hawor/dl_hawor.log 2>&1
      touch "$HF/.hawor_done"
    fi
    if [ ! -f "$HF/.mano_done" ]; then
      echo "[weights] hf download MANO (lithiumice/models_hub)"
      hf download lithiumice/models_hub 4_SMPLhub/MANO/pkl/MANO_RIGHT.pkl 4_SMPLhub/MANO/pkl/MANO_LEFT.pkl --local-dir "$HF/mano" >/data/overnight-tests/hawor/dl_mano.log 2>&1
      touch "$HF/.mano_done"
    fi
    mkdir -p weights/external weights/hawor/checkpoints thirdparty/Metric3D/weights _DATA/data/mano _DATA/data_left/mano_left
    cp -f "$HF/hawor/external/detector.pt"                      weights/external/
    cp -f "$HF/hawor/external/droid.pth"                        weights/external/
    cp -f "$HF/hawor/hawor/checkpoints/hawor.ckpt"             weights/hawor/checkpoints/
    cp -f "$HF/hawor/hawor/checkpoints/infiller.pt"            weights/hawor/checkpoints/
    cp -f "$HF/hawor/hawor/model_config.yaml"                  weights/hawor/
    cp -f "$HF/hawor/external/metric_depth_vit_large_800k.pth" thirdparty/Metric3D/weights/
    cp -f "$HF/mano/4_SMPLhub/MANO/pkl/MANO_RIGHT.pkl"         _DATA/data/mano/MANO_RIGHT.pkl
    cp -f "$HF/mano/4_SMPLhub/MANO/pkl/MANO_LEFT.pkl"          _DATA/data_left/mano_left/MANO_LEFT.pkl
    # STAGE 3b: neutralize pytorch3d hand-mask render (SLAM runs unmasked on ROCm)
    python - <<PY
import re
f="scripts/scripts_test_video/hawor_video.py"; s=open(f).read()
if "ROCm stub" not in s:
    s=s.replace("from lib.vis.renderer import Renderer","Renderer=None  # ROCm stub")
    s=re.sub(r"renderer = Renderer\(.*?max_faces_per_bin=max_faces_per_bin\)","renderer=None  # ROCm disabled",s,flags=re.S)
    s=re.sub(r"vertices = outputs\[\"vertices\"\]\[0\]\.cpu\(\).*?model_masks\[frame_ck\[img_i\]\] \+= mask","pass  # ROCm: hand-mask render skipped",s,flags=re.S)
    open(f,"w").write(s); print("[patch] hawor_video.py pytorch3d neutralized")
else: print("[patch] hawor_video.py already patched")
PY
    ls -la weights/external weights/hawor/checkpoints _DATA/data/mano _DATA/data_left/mano_left 2>&1 | grep -E "\.pt|\.pth|\.ckpt|\.pkl" | head
    echo WEIGHTS_READY
  '
}

run_one() {
  local mp4="$1" focal="$2" id="$3"
  local dst="$RUNS/$id.mp4"
  cp -f "$mp4" "$dst"
  local fopt=""; [ -n "$focal" ] && fopt="--img_focal $focal"
  echo "[run] $id  focal=$focal"
  docker exec "$CNAME" bash -lc "cp -f $WORK/hawor_infer_egodex.py /opt/HaWoR/ && cd /opt/HaWoR && python hawor_infer_egodex.py --video_path '$dst' $fopt --hawor_repo /opt/HaWoR"
}

MP4=""; FOCAL=""; ID=""
while [ $# -gt 0 ]; do case "$1" in
  --setup) shift;;
  --video) MP4="$2"; shift 2;;
  --focal) FOCAL="$2"; shift 2;;
  --id) ID="$2"; shift 2;;
  *) shift;;
esac; done

ensure_container
setup_weights
if [ -n "$MP4" ]; then
  [ -z "$ID" ] && ID=$(basename "$MP4" .mp4)
  run_one "$MP4" "$FOCAL" "$ID"
fi
