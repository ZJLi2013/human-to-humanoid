#!/usr/bin/env bash
# M1-a step 1: DOWNLOAD ONLY (no video extraction).
#  - SSv2 raw videos: morpheushoc/something-something-v2 (canonical 20bn split archive, ~19.5GB,
#    monolithic -> must be fetched whole; extraction is done selectively in step 2 for ~10k ids).
#  - SSv2 VITRA-1M annotations: VITRA-VLA/VITRA-1M/ssv2.tar.gz (~2.33GB) -> extracted (small).
# Runs inside docker (root) so it can write /data. Resumable (curl -C -).
set -uo pipefail

IMAGE="${IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
DATA_DIR="${DATA_DIR:-/data}"
ROOT="${DATA_DIR}/overnight-tests/vitra/data_ssv2_10k"
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"

docker run --rm -i \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  -e ROOT="${ROOT}" \
  "${IMAGE}" bash -s <<'INNER'
set -uo pipefail
TOK=$(cat /root/.hf_token | tr -d '[:space:]')
VID_BASE=https://huggingface.co/datasets/morpheushoc/something-something-v2/resolve/main
ANN_BASE=https://huggingface.co/datasets/VITRA-VLA/VITRA-1M/resolve/main
RAW="${ROOT}/_raw_video"      # monolithic archive parts (kept, reusable for M2)
ANN="${ROOT}/annotation"      # extracted VITRA-1M ssv2 annotations
mkdir -p "${RAW}" "${ANN}"

echo "=== [1/3] download 20 video archive parts (~19.5GB, resumable) ==="
for i in $(seq -w 0 19); do
  f="20bn-something-something-v2-${i}"
  if [ -s "${RAW}/${f}" ] && [ "$(stat -c%s "${RAW}/${f}")" -ge 900000000 -o "${i}" = "19" ]; then
    echo "skip ${f}"; continue
  fi
  echo "-- ${f} --"
  curl -fL --retry 5 -C - -H "Authorization: Bearer ${TOK}" -o "${RAW}/${f}" "${VID_BASE}/videos/${f}" \
    || { echo "DOWNLOAD_FAIL ${f}"; exit 2; }
done

echo "=== [2/3] download SSv2 annotation ssv2.tar.gz (~2.33GB) ==="
curl -fL --retry 5 -C - -H "Authorization: Bearer ${TOK}" \
  -o "${RAW}/ssv2.tar.gz" "${ANN_BASE}/ssv2.tar.gz" || { echo "ANN_DL_FAIL"; exit 3; }
curl -fL --retry 5 -H "Authorization: Bearer ${TOK}" \
  -o "${ANN}/ssv2_angle_statistics.json" "${ANN_BASE}/statistics/ssv2_angle_statistics.json" || echo "warn: stats"

echo "=== [3/3] extract annotation only (small) ==="
tar -xzf "${RAW}/ssv2.tar.gz" -C "${ANN}"
echo "=== annotation layout (top 40) ==="
find "${ANN}" -maxdepth 3 | head -40
echo "=== sizes ==="; du -sh "${RAW}" "${ANN}"
echo "DOWNLOAD_ONLY_DONE (videos NOT extracted; selective extract in step 2)"
INNER
echo "EXIT=$?"
