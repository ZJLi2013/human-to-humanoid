#!/usr/bin/env bash
# Fast M1-c launch using the pre-baked image (vitra-rocm:m1c): deps + VITRA + decord patch
# are already inside, so this skips the ~7min pip install. Only generates the config + runs.
set -uo pipefail

RUN_ID="${RUN_ID:-m1c_ssv2_10k}"
IMAGE="${IMAGE:-vitra-rocm:m1c}"
DATA_DIR="${DATA_DIR:-/data}"
WORK="${DATA_DIR}/overnight-tests/vitra"
OUT="${WORK}/outputs_rocm/${RUN_ID}"
CACHE="${DATA_DIR}/cache/huggingface"
DATA_ROOT="${WORK}/data_ssv2_10k"
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"
NPROC="${NPROC:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"

echo "=== VITRA M1-c FAST ${RUN_ID} (image=${IMAGE}, nproc=${NPROC}, max_steps=${MAX_STEPS}) ==="

docker run --rm -i \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e RUN_ID="${RUN_ID}" -e OUT="${OUT}" -e DATA_ROOT="${DATA_ROOT}" -e NPROC="${NPROC}" -e MAX_STEPS="${MAX_STEPS}" \
  -e HF_HOME="${CACHE}" -e TRANSFORMERS_CACHE="${CACHE}" \
  -e WANDB_MODE=offline -e WANDB_API_KEY=local-offline -e WANDB_DIR="${OUT}" \
  -e NCCL_DEBUG=WARN -e TOKENIZERS_PARALLELISM=false \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_IB_DISABLE=1 \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  "${IMAGE}" bash -s <<'INNER'
set -uo pipefail
cd /opt/VITRA
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
mkdir -p "${OUT}"

echo "--- sanity: torch/gpu + baked deps ---"
python - <<'PY'
import torch, utils3d.numpy as n
print("torch", torch.__version__, "hip", getattr(torch.version,"hip",None),
      "n_gpu", torch.cuda.device_count(), "image_uv", hasattr(n,"image_uv"))
PY

echo "--- generate ssv2_10k.json ---"
DATA_ROOT="${DATA_ROOT}" OUT="${OUT}" MAX_STEPS="${MAX_STEPS}" python - <<'PY'
import json, os
c=json.load(open("vitra/configs/human_pretrain.json"))
OUT=os.environ["OUT"]; DR=os.environ["DATA_ROOT"]; MS=int(os.environ["MAX_STEPS"])
c["batch_size"]=8; c["total_batch_size"]=64; c["save_steps"]=250
c["output_root"]=os.path.join(OUT,"checkpoints")
c["log_root"]=os.path.join(OUT,"logs")
c["cache_root"]=os.path.join(OUT,"cache")
c["wandb_entity"]=""
c["trainer"]["max_steps"]=MS
c["trainer"]["llm_freeze_step"]=200
c["train_dataset"]["data_root_dir"]=DR
c["train_dataset"]["data_mix"]="ssv2"
c["train_dataset"]["num_workers"]=4
json.dump(c, open("vitra/configs/ssv2_10k.json","w"), indent=2)
print("wrote ssv2_10k.json max_steps=%d data_root=%s"%(MS,DR))
PY

echo "--- torchrun FSDP (${NPROC} GPUs) ---"
torchrun --standalone --nproc_per_node="${NPROC}" scripts/train.py --config vitra/configs/ssv2_10k.json
RC=$?
echo "TORCHRUN_EXIT=$RC"
[ "$RC" = "0" ] && echo "M1C_TRAIN_DONE" || echo "M1C_TRAIN_FAIL"
INNER
echo "EXIT_CODE=$?"
