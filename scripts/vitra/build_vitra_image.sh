#!/usr/bin/env bash
# Bake a ready-to-run VITRA-on-ROCm image so runs skip the ~7min dep install.
# Contents: base rocm/pytorch 2.9.1 + all training deps (utils3d pinned @d790d33)
# + microsoft/VITRA cloned to /opt/VITRA (editable, no-deps) + decord num_threads=1 patch.
# ROCm-specific env (HSA_NO_SCRATCH_RECLAIM) stays a runtime flag, not baked.
set -uo pipefail
BASE="${BASE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
TAG="${TAG:-vitra-rocm:m1c}"
CNAME=vitra_img_build

echo "=== build ${TAG} from ${BASE} ==="
docker rm -f "$CNAME" 2>/dev/null || true
docker run --name "$CNAME" "$BASE" bash -c '
set -e
cd /opt
git clone --depth 1 https://github.com/microsoft/VITRA.git
cd VITRA
git config --global --add safe.directory /opt/VITRA
python -m pip install --no-input -q --no-deps -e .
python -m pip install --no-input -q \
  "transformers==4.47.1" accelerate safetensors "numpy<2" pillow scipy \
  diffusers "timm==0.9.10" einops sentencepiece "tokenizers>=0.21" "peft==0.11.1" \
  hydra-core draccus wandb rich tqdm tensorboard tensorboardX pyyaml \
  opencv-python-headless decord scikit-image roma trimesh ffmpeg-python \
  imageio imageio-ffmpeg pympler json-numpy jsonlines matplotlib omegaconf \
  "huggingface_hub>=0.24" "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@d790d33"
# decord threaded-decoder concurrency fix (platform-agnostic)
sed -i "s/decord.VideoReader(name)/decord.VideoReader(name, num_threads=1)/g" vitra/datasets/video_utils.py
grep -n "decord.VideoReader" vitra/datasets/video_utils.py
python -c "import vitra, utils3d.numpy as n; print(\"BAKE_OK import vitra + image_uv=\", hasattr(n,\"image_uv\"))"
'
RC=$?
echo "BUILD_INNER_RC=$RC"
[ "$RC" = "0" ] || { echo "BUILD_FAIL"; exit 1; }
echo "=== commit ==="
docker commit -c 'ENV HSA_NO_SCRATCH_RECLAIM=1' -c 'WORKDIR /opt/VITRA' "$CNAME" "$TAG"
docker rm "$CNAME"
docker images "$TAG"
echo "IMAGE_SAVED ${TAG}"
