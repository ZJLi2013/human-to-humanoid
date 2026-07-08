#!/usr/bin/env bash
# Fetch EgoDex test set (16 GB, Apple CC-by-NC-ND) as held-out GT for HaWoR accuracy eval.
# Idempotent: skips download/unzip if already present. /data/overnight-tests is root-owned,
# so we mint a self-owned subdir via a throwaway alpine container (we're in the docker group).
#
#   run:  bash fetch_egodex.sh            # -> /data/overnight-tests/egodex/test/<task>/<i>.{hdf5,mp4}
#   env:  EGODEX_DIR (default /data/overnight-tests/egodex)  SPLIT (default test)
set -uo pipefail

EGODEX_DIR="${EGODEX_DIR:-/data/overnight-tests/egodex}"
SPLIT="${SPLIT:-test}"
URL="https://ml-site.cdn-apple.com/datasets/egodex/${SPLIT}.zip"
ZIP="$EGODEX_DIR/${SPLIT}.zip"
OUT="$EGODEX_DIR/${SPLIT}"

U=$(id -u); G=$(id -g)
if [ ! -w "$(dirname "$EGODEX_DIR")" ] && [ ! -d "$EGODEX_DIR" ]; then
  echo "[setup] minting self-owned $EGODEX_DIR via alpine container"
  docker run --rm -v /data:/data alpine mkdir -p "$EGODEX_DIR"
  docker run --rm -v /data:/data alpine chown -R "$U:$G" "$EGODEX_DIR"
fi
mkdir -p "$EGODEX_DIR"

if [ -d "$OUT" ] && [ "$(ls "$OUT" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[skip] $OUT already populated ($(ls "$OUT" | wc -l) tasks)"; du -sh "$OUT"; exit 0
fi

if [ ! -f "$ZIP" ]; then
  echo "[download] $URL -> $ZIP  ($(date))"
  curl -fL "$URL" -o "$ZIP.part" && mv "$ZIP.part" "$ZIP"
  echo "[download] CURL_RC=$? size=$(du -h "$ZIP" | cut -f1)"
else
  echo "[skip] zip present: $(du -h "$ZIP" | cut -f1)"
fi

echo "[unzip] -> $EGODEX_DIR (auto-detect top-level)"
unzip -o -q "$ZIP" -d "$EGODEX_DIR" && echo "UNZIP_OK" || { echo "UNZIP_FAIL"; exit 1; }
# EgoDex zip may or may not nest a top-level split dir; locate the folder holding <task>/<i>.hdf5
REAL=$(dirname "$(find "$EGODEX_DIR" -maxdepth 3 -name '*.hdf5' 2>/dev/null | head -1)")   # -> .../<task>
REAL=$(dirname "$REAL")                                                                    # -> split root
echo "[done] split_root=$REAL  tasks=$(ls "$REAL" 2>/dev/null | wc -l)  size=$(du -sh "$EGODEX_DIR" | cut -f1)  ($(date))"
ls "$REAL" 2>/dev/null | head
