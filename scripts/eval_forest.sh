#!/usr/bin/env bash
# Evaluate the released PoinTTT checkpoint on the FOR-instanceV2 test split (28 plots, 1 GPU).
#
#   bash scripts/eval_forest.sh [checkpoint]        # default: checkpoints/pointttt_forainetv2.pth
#
# Environment variables (all optional):
#   PYTHON      interpreter of the environment (default: ./env/bin/python if it exists, else python)
#   DATA_ROOT   FOR-instanceV2 root with test_data/ and meta_data/test_list.txt (default: data/forainetv2)
#   OUT         output directory (default: exp/forainetv2_eval); the full log is OUT/train.log
#   CUDA_VISIBLE_DEVICES  selects the GPU
#
# Expected on one NVIDIA B200 (torch 2.7.1+cu128):
#   Val result: mIoU/mAcc/allAcc 0.8789/0.9188/0.9601.
#   Class_0-ground Result: iou/accuracy 0.9928/0.9956
#   Class_1-wood Result: iou/accuracy 0.6929/0.7786
#   Class_2-leaf Result: iou/accuracy 0.9511/0.9820
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
CKPT=$(abspath "${1:-$REPO/checkpoints/pointttt_forainetv2.pth}")
DATA_ROOT=$(abspath "${DATA_ROOT:-$REPO/data/forainetv2}")
OUT=$(abspath "${OUT:-$REPO/exp/forainetv2_eval}")
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=python; fi
fi

# the --options parser of tools/train.py splits values at commas and drops blanks
for p in "$REPO" "$CKPT" "$DATA_ROOT" "$OUT"; do
  case "$p" in
    *[[:space:]]*|*,*|*\[*|*\]*|*\(*|*\)*|*\'*|*\"*)
      echo "paths with blanks, commas, brackets or quotes are not supported: $p" >&2; exit 2 ;;
  esac
done
[ -s "$CKPT" ] || { echo "checkpoint not found: $CKPT (see README, Checkpoints)" >&2; exit 1; }
LIST=$DATA_ROOT/meta_data/test_list.txt
[ -s "$LIST" ] || { echo "missing $LIST: run scripts/prepare_forest.sh (see README, Data)" >&2; exit 1; }
n=0
while read -r name || [ -n "$name" ]; do
  [ -n "$name" ] || continue
  [ -s "$DATA_ROOT/test_data/$name.ply" ] || { echo "missing $DATA_ROOT/test_data/$name.ply: run scripts/prepare_forest.sh" >&2; exit 1; }
  n=$((n + 1))
done < "$LIST"
[ "$n" -eq 28 ] || echo "WARNING: $LIST lists $n plots; the official test split has 28" >&2

cd "$REPO"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONNOUSERSITE=1
unset SPCONV_ALGO   # the reported numbers use spconv's default convolution algorithm
mkdir -p "$OUT"
"$PYTHON" tools/train.py --config-file configs/forainetv2/pointttt-forainetv2-eval.py --num-gpus 1 \
    --options weight="$CKPT" save_path="$OUT" data.val.data_root="$DATA_ROOT"
echo
grep -E "Val result|Class_[0-9]+-" "$OUT/train.log" | tail -4 | sed -E 's/^\[[^]]*\] //'
