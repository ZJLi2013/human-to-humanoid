#!/usr/bin/env bash
# feature8 Exp-8.0: run vitra_action_decode.py --mode probe inside vitra-rocm image.
# Dumps VITRA ActionFeature/StateFeature layout + normalizer dims so we can decode the
# action chunk (wrist 6D + finger joints) without hardcoding dims. GPU not required for probe.
set -uo pipefail

IMAGE="${IMAGE:-vitra-rocm:m1c}"
DATA_DIR="${DATA_DIR:-/data}"
REPO="${REPO:-/data/zhengjli/human-to-humanoid}"
VITRA_REPO="${VITRA_REPO:-/data/overnight-tests/vitra/VITRA}"
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"
CACHE="${DATA_DIR}/cache/huggingface"
MODE="${MODE:-probe}"          # probe | decode
IMG_ARG="${IMG_ARG:-}"         # decode: optional image path
OUT="${OUT:-/data/zhengjli/human-to-humanoid/_out/vitra_probe}"

echo "=== VITRA action ${MODE} (Exp-8.0) IMAGE=${IMAGE} ==="
docker run --rm -i \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HF_HOME="${CACHE}" -e TRANSFORMERS_CACHE="${CACHE}" \
  -e REPO="${REPO}" -e VITRA_REPO="${VITRA_REPO}" -e MODE="${MODE}" \
  -e IMG_ARG="${IMG_ARG}" -e OUT="${OUT}" \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  -w "${VITRA_REPO}" \
  "${IMAGE}" bash -s <<'INNER'
set -uo pipefail
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
python - <<'PY' || echo "HF_LOGIN_WARN"
import os
from huggingface_hub import login, whoami
try:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("hf whoami:", whoami()["name"])
except Exception as e:
    print("hf login WARN:", repr(e))
PY

mkdir -p "${OUT}"
ARGS="--vitra_repo ${VITRA_REPO} --mode ${MODE}"
if [ "${MODE}" = "decode" ]; then
  ARGS="${ARGS} --out ${OUT}/hand_traj.npz"
  [ -n "${IMG_ARG}" ] && ARGS="${ARGS} --image ${IMG_ARG}"
fi
echo "--- run vitra_action_decode.py ${ARGS} ---"
python "${REPO}/scripts/human_video_factory/vitra_action_decode.py" ${ARGS} 2>&1 | tee "${OUT}/${MODE}.log"
echo "PROBE_EXIT=${PIPESTATUS[0]}"
INNER
echo "EXIT_CODE=$?"
