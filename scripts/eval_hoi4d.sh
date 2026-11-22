#!/usr/bin/env bash
# Evaluate the released PoinTTT checkpoint on the HOI4D test set (892 sequences) and print pptr_mIoU.
#
#   bash scripts/eval_hoi4d.sh [checkpoint]         # default: checkpoints/pointttt_hoi4d.pth
#
# Environment variables (all optional):
#   NUM_GPUS     1, 2 or 4; must divide the 892 test sequences (default: 4, 2 or 1, the most the visible
#                GPUs allow). The reported number was measured with NUM_GPUS=4.
#   SEQ_PER_GPU  sequences per GPU and batch, 1-8 (default 8; the batch buffers alone take about 18 GB of
#                GPU memory, use 2-4 on 24-40 GB GPUs)
#   NUM_WORKER   data loader workers in total (default 2 per GPU; each holds about 3 GB of prefetched data)
#   PYTHON       interpreter of the environment (default: ./env/bin/python if it exists, else python)
#   DATA_ROOT    directory with test1_right.h5 and test2_right.h5 (default: data/hoi4d)
#   OUT          output directory (default: exp/hoi4d_eval); the full log is OUT/train.log
#
# Expected with NUM_GPUS=4 SEQ_PER_GPU=8 on NVIDIA B200 (torch 2.7.1+cu128), about 75 minutes:
#   [pptr_score] block=1 ts=<date> raw_mIoU=0.3388 pptr_mIoU=0.4257 active_classes=39 raw_mAcc=0.4442 raw_allAcc=0.7803
# Other GPU counts or batch sizes change the random parts of the evaluation (expect about +-0.0005).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
CKPT=$(abspath "${1:-$REPO/checkpoints/pointttt_hoi4d.pth}")
DATA_ROOT=$(abspath "${DATA_ROOT:-$REPO/data/hoi4d}")
OUT=$(abspath "${OUT:-$REPO/exp/hoi4d_eval}")
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=python; fi
fi
if [ -z "${NUM_GPUS:-}" ]; then
  NGPU=$(PYTHONNOUSERSITE=1 "$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1 || true)
  case "$NGPU" in ''|*[!0-9]*) NGPU=0 ;; esac
  if [ "$NGPU" -ge 4 ]; then NUM_GPUS=4; elif [ "$NGPU" -ge 2 ]; then NUM_GPUS=2; else NUM_GPUS=1; fi
fi
SEQ_PER_GPU=${SEQ_PER_GPU:-8}
NUM_WORKER=${NUM_WORKER:-$((2 * NUM_GPUS))}
case "$NUM_GPUS" in 1|2|4) ;; *) echo "NUM_GPUS=$NUM_GPUS: use 1, 2 or 4 (must divide the 892 test sequences)" >&2; exit 2 ;; esac
case "$SEQ_PER_GPU" in [1-8]) ;; *) echo "SEQ_PER_GPU=$SEQ_PER_GPU: use 1-8" >&2; exit 2 ;; esac
case "$NUM_WORKER" in ''|*[!0-9]*) echo "NUM_WORKER=$NUM_WORKER is not a number" >&2; exit 2 ;; esac

# the --options parser of tools/train.py splits values at commas and drops blanks
for p in "$REPO" "$CKPT" "$DATA_ROOT" "$OUT"; do
  case "$p" in
    *[[:space:]]*|*,*|*\[*|*\]*|*\(*|*\)*|*\'*|*\"*)
      echo "paths with blanks, commas, brackets or quotes are not supported: $p" >&2; exit 2 ;;
  esac
done
[ -s "$CKPT" ] || { echo "checkpoint not found: $CKPT (see README, Checkpoints)" >&2; exit 1; }
for f in test1_right test2_right; do
  [ -s "$DATA_ROOT/$f.h5" ] || { echo "missing $DATA_ROOT/$f.h5 (see README, Data)" >&2; exit 1; }
done

cd "$REPO"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE
export SPCONV_ALGO=Native   # no int32 overflow at large batches; the reported number uses it
ulimit -n 65536 2>/dev/null || true
BS=$((SEQ_PER_GPU * NUM_GPUS))
mkdir -p "$OUT"
echo "HOI4D evaluation: $NUM_GPUS GPU(s) x $SEQ_PER_GPU sequences, $NUM_WORKER loader workers, output in $OUT"
"$PYTHON" tools/train.py --config-file configs/hoi4d/pointttt-hoi4d-eval.py --num-gpus "$NUM_GPUS" \
    --options weight="$CKPT" save_path="$OUT" data.val.data_root="$DATA_ROOT" \
    batch_size=$BS batch_size_val=$BS num_worker=$NUM_WORKER
echo
"$PYTHON" scripts/pptr_miou.py "$OUT/train.log" --latest
