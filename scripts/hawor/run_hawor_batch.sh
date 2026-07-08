#!/usr/bin/env bash
# Batch HaWoR front-end over a diverse set of EgoDex clips (round-robin across tasks) into the
# already-running `hawor_run` container. Idempotent: skips clips whose world_space_res.pth exists.
# EgoDex camera intrinsic is constant across the dataset -> focal hardcoded (README: "always the
# same in every file"); center (960,540) for 1920x1080.
#
#   bash run_hawor_batch.sh [N_PER_TASK] [FOCAL]     # default 3 clips/task, focal 736.6339
set -uo pipefail

EGODEX="${EGODEX:-/data/overnight-tests/egodex/test}"
WORK="${WORK:-/data/overnight-tests/hawor}"
CNAME="${CNAME:-hawor_run}"
RUNS="$WORK/runs"
N_PER_TASK="${1:-1}"
FOCAL="${2:-736.6339}"
MAX_TOTAL="${MAX_TOTAL:-40}"

mkdir -p "$RUNS"
mapfile -t TASKS < <(ls "$EGODEX")
echo "[batch] ${#TASKS[@]} tasks, $N_PER_TASK clips/task, MAX_TOTAL=$MAX_TOTAL, focal=$FOCAL  ($(date))"

done_n=0; skip_n=0; fail_n=0; i=0
for task in "${TASKS[@]}"; do
  [ "$i" -ge "$MAX_TOTAL" ] && break
  k=0
  for mp4 in $(ls "$EGODEX/$task"/*.mp4 2>/dev/null | sort -V); do
    [ "$k" -ge "$N_PER_TASK" ] && break
    k=$((k+1)); i=$((i+1))
    id="${task}__$(basename "$mp4" .mp4)"
    seq="$RUNS/$id"
    if [ -f "$seq/world_space_res.pth" ]; then
      echo "  [skip] $id (done)"; skip_n=$((skip_n+1)); continue
    fi
    cp -f "$mp4" "$RUNS/$id.mp4"
    echo "  [$i] $id ..."
    if docker exec "$CNAME" bash -lc "cp -f $WORK/hawor_infer_egodex.py /opt/HaWoR/ && cd /opt/HaWoR && python hawor_infer_egodex.py --video_path '$RUNS/$id.mp4' --img_focal $FOCAL --hawor_repo /opt/HaWoR" > "$RUNS/$id.log" 2>&1; then
      if [ -f "$seq/world_space_res.pth" ]; then
        done_n=$((done_n+1)); echo "     OK -> $seq"
      else
        fail_n=$((fail_n+1)); echo "     FAIL (no output); tail:"; tail -3 "$RUNS/$id.log"
      fi
    else
      fail_n=$((fail_n+1)); echo "     FAIL (rc); tail:"; tail -3 "$RUNS/$id.log"
    fi
  done
done
echo "[batch] done=$done_n skip=$skip_n fail=$fail_n  ($(date))"
