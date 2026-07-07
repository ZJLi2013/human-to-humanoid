#!/usr/bin/env bash
# M1-a step 2 (v2, idempotent+robust): verify/repair archive parts -> crop 10k episodes ->
# extract ONLY selected webm via Python tarfile streaming filter (precise, no tar -T ambiguity).
set -uo pipefail
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROOT=/data/overnight-tests/vitra/data_ssv2_10k
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"
N_EP="${N_EP:-10000}"; SEED="${SEED:-42}"

docker run --rm -i -v /data:/data -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  -e ROOT="$ROOT" -e N_EP="$N_EP" -e SEED="$SEED" "$IMAGE" bash -s <<'INNER'
set -uo pipefail
TOK=$(cat /root/.hf_token | tr -d '[:space:]')
VID_BASE=https://huggingface.co/datasets/morpheushoc/something-something-v2/resolve/main/videos
RAW="$ROOT/_raw_video"
SRC_ANN="$ROOT/annotation"; DST_ANN="$ROOT/Annotation"
VID_ROOT="$ROOT/Video/Somethingsomething-v2_root"

echo "=== [1/4] verify part sizes (00-18 == 1000000000, 19 == 444975475); re-download if off ==="
for i in $(seq -w 0 19); do
  f="20bn-something-something-v2-${i}"; exp=1000000000; [ "$i" = "19" ] && exp=444975475
  cur=$(stat -c%s "$RAW/$f" 2>/dev/null || echo 0)
  if [ "$cur" != "$exp" ]; then
    echo "  repair $f (have $cur want $exp)"
    curl -fL --retry 5 -H "Authorization: Bearer ${TOK}" -o "$RAW/$f" "${VID_BASE}/${f}" || { echo "REDL_FAIL $f"; exit 2; }
    cur=$(stat -c%s "$RAW/$f"); [ "$cur" = "$exp" ] || { echo "SIZE_STILL_BAD $f=$cur"; exit 2; }
  else echo "  ok $f ($cur)"; fi
done

echo "=== [2/4] select ${N_EP} episodes + write cropped index + member list ==="
mkdir -p "$DST_ANN/ssv2" "$DST_ANN/statistics" "$VID_ROOT"
python3 - <<'PY'
import numpy as np, os
ROOT=os.environ["ROOT"]; N=int(os.environ["N_EP"]); SEED=int(os.environ["SEED"])
z=np.load(os.path.join(ROOT,"annotation/ssv2/episode_frame_index.npz"), allow_pickle=True)
ifp=z["index_frame_pair"]; i2e=z["index_to_episode_id"]; n_ep=len(i2e)
rng=np.random.default_rng(SEED)
sel=np.sort(rng.choice(n_ep, size=min(N,n_ep), replace=False))
ifp_crop=ifp[np.isin(ifp[:,0], sel)]
vids=sorted({ str(i2e[i]).split('_')[1] for i in sel })
np.savez(os.path.join(ROOT,"Annotation/ssv2/episode_frame_index.npz"),
         index_frame_pair=ifp_crop, index_to_episode_id=i2e)
open("/tmp/members.txt","w").write("\n".join(f"20bn-something-something-v2/{v}.webm" for v in vids)+"\n")
print(f"selected_episodes={len(sel)} frames_after_crop={len(ifp_crop)} (was {len(ifp)}) unique_videos={len(vids)}")
PY

echo "=== [3/4] link episodic_annotations + copy statistics ==="
[ -e "$DST_ANN/ssv2/episodic_annotations" ] || ln -s "$SRC_ANN/ssv2/episodic_annotations" "$DST_ANN/ssv2/episodic_annotations"
cp -f "$SRC_ANN/ssv2_angle_statistics.json" "$DST_ANN/statistics/ssv2_angle_statistics.json"

echo "=== [4/4] clear stale videos + stream-extract ONLY selected (python tarfile) ==="
find "$VID_ROOT" -name '*.webm' -delete 2>/dev/null || true
# NOTE: write extractor to a FILE (cannot use `python3 - <<HEREDOC` while also piping the
# archive to stdin — the heredoc would hijack stdin).
cat > /tmp/extract.py <<'PY'
import tarfile, sys, os
ROOT=os.environ["ROOT"]; VID=os.path.join(ROOT,"Video/Somethingsomething-v2_root")
want=set(l.strip() for l in open("/tmp/members.txt") if l.strip())
found=0; total=len(want)
tf=tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")
for m in tf:
    if m.name in want and m.isfile():
        m.name=os.path.basename(m.name)
        tf.extract(m, VID)
        found+=1
        if found % 1000 == 0: print(f"  extracted {found}/{total}", flush=True)
        if found==total: break
print(f"EXTRACTED={found}/{total}", flush=True)
PY
cat "$RAW"/20bn-something-something-v2-?? | python3 /tmp/extract.py

echo "=== verify ==="
want=$(wc -l < /tmp/members.txt); got=$(find "$VID_ROOT" -name '*.webm' | wc -l)
echo "members_wanted=$want webm_extracted=$got"; du -sh "$VID_ROOT"
ls "$VID_ROOT" | head -3
[ "$want" = "$got" ] && echo "CROP_10K_DONE" || echo "CROP_MISMATCH"
INNER
echo "EXIT=$?"
