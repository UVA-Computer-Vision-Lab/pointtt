#!/usr/bin/env bash
# Evaluate a PoinTTT checkpoint with an evaluation config (tools/test.py).
#
#   bash scripts/test.sh -d forainetv2              # checkpoints/pointttt_forainetv2.pth on the 28 test plots, 1 GPU
#   bash scripts/test.sh -d hoi4d -g 4              # checkpoints/pointttt_hoi4d.pth on the 892 test sequences
#
# Options:
#   -d DATASET  forainetv2 or hoi4d (required)
#   -c CONFIG   config name in configs/DATASET/ (default: pointttt-DATASET-eval)
#   -w FILE     checkpoint (default: checkpoints/pointttt_DATASET.pth)
#   -g GPUS     forainetv2: 1. hoi4d: 1, 2 or 4, must divide the 892 test sequences (default: 4, 2 or 1,
#               the most the visible GPUs allow; the reported number was measured with 4)
#   -n DIR      output directory (default: exp/DATASET_eval); the full log is DIR/train.log
#   -p PYTHON   interpreter of the environment (default: ./env/bin/python if it exists, else python)
#   -o "K=V .." extra config overrides
# Environment variables (all optional): PYTHON, DATA_ROOT (directory with the test data, default
# data/DATASET), CUDA_VISIBLE_DEVICES; for hoi4d also SEQ_PER_GPU (sequences per GPU and batch, 1-8,
# default 8; the batch buffers alone take about 18 GB of GPU memory, use 2-4 on 24-40 GB GPUs) and
# NUM_WORKER (loader workers in total, default 2 per GPU; each holds about 3 GB of prefetched data).
# NUM_GPUS and OUT are accepted in place of -g and -n. DRY_RUN=1 prints the command instead of running it.
#
# Expected on NVIDIA B200 (torch 2.7.1+cu128):
#   forainetv2, 1 GPU, a few minutes:
#     Val result: mIoU/mAcc/allAcc 0.8789/0.9188/0.9601.
#     Class_0-ground Result: iou/accuracy 0.9928/0.9956
#     Class_1-wood Result: iou/accuracy 0.6929/0.7786
#     Class_2-leaf Result: iou/accuracy 0.9511/0.9820
#   hoi4d, -g 4 with SEQ_PER_GPU=8, about 75 minutes:
#     [pptr_score] block=1 ts=<date> raw_mIoU=0.3388 pptr_mIoU=0.4257 active_classes=39 raw_mAcc=0.4442 raw_allAcc=0.7803
#   Other GPU counts or batch sizes change the random parts of the HOI4D evaluation (about +-0.0005).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
usage() { awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}" >&2; exit "${1:-2}"; }

# stop early, with a hint, when fewer GPUs are visible than requested
check_gpus() {  # check_gpus N HINT
  local vis
  vis=$(PYTHONNOUSERSITE=1 "$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1 || true)
  case "$vis" in ''|*[!0-9]*) return 0 ;; esac   # cannot tell: let the run report it
  if [ "$vis" -lt "$1" ]; then
    echo "$1 GPU(s) requested, but $vis visible (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset})$2" >&2
    exit 1
  fi
}

DATASET="" CONFIG="" CKPT="" NGPU=${NUM_GPUS:-} OUT=${OUT:-} EXTRA=""
while getopts "d:c:w:g:n:p:o:h" opt; do
  case $opt in
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    w) CKPT=$OPTARG ;;
    g) NGPU=$OPTARG ;;
    n) OUT=$OPTARG ;;
    p) PYTHON=$OPTARG ;;
    o) EXTRA=$OPTARG ;;
    h) usage 0 ;;
    *) usage ;;
  esac
done
[ "$OPTIND" -gt $# ] || { echo "unexpected argument: ${!OPTIND}" >&2; usage; }
case "$DATASET" in
  forainetv2|hoi4d) ;;
  "") echo "-d is required: forainetv2 or hoi4d" >&2; usage ;;
  *) echo "-d $DATASET: use forainetv2 or hoi4d" >&2; exit 2 ;;
esac
CONFIG=${CONFIG:-pointttt-$DATASET-eval}
CFG_FILE=$REPO/configs/$DATASET/$CONFIG.py
[ -f "$CFG_FILE" ] || { echo "no such config: configs/$DATASET/$CONFIG.py" >&2; exit 2; }
CKPT=$(abspath "${CKPT:-$REPO/checkpoints/pointttt_$DATASET.pth}")
DATA_ROOT=$(abspath "${DATA_ROOT:-$REPO/data/$DATASET}")
OUT=$(abspath "${OUT:-$REPO/exp/${DATASET}_eval}")
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=python; fi
fi

# the --options parser splits values at commas and drops blanks
for p in "$REPO" "$CKPT" "$DATA_ROOT" "$OUT"; do
  case "$p" in
    *[[:space:]]*|*,*|*\[*|*\]*|*\(*|*\)*|*\'*|*\"*)
      echo "paths with blanks, commas, brackets or quotes are not supported: $p" >&2; exit 2 ;;
  esac
done
[ -s "$CKPT" ] || { echo "checkpoint not found: $CKPT (see README, Pretrained Checkpoints)" >&2; exit 1; }

export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONNOUSERSITE=1
OPTS="weight=$CKPT save_path=$OUT data.val.data_root=$DATA_ROOT"

if [ "$DATASET" = forainetv2 ]; then
  NGPU=${NGPU:-1}
  [ "$NGPU" = 1 ] || { echo "-g $NGPU: the FOR-instanceV2 evaluation runs on 1 GPU (one plot per batch)" >&2; exit 2; }
  LIST=$DATA_ROOT/meta_data/test_list.txt
  [ -s "$LIST" ] || { echo "missing $LIST: run scripts/prepare_forest.sh (see README, Datasets)" >&2; exit 1; }
  n=0
  while read -r name || [ -n "$name" ]; do
    [ -n "$name" ] || continue
    [ -s "$DATA_ROOT/test_data/$name.ply" ] || { echo "missing $DATA_ROOT/test_data/$name.ply: run scripts/prepare_forest.sh" >&2; exit 1; }
    n=$((n + 1))
  done < "$LIST"
  [ "$n" -eq 28 ] || echo "WARNING: $LIST lists $n plots; the official test split has 28" >&2
  unset SPCONV_ALGO   # the reported numbers use spconv's default convolution algorithm
else
  if [ -z "$NGPU" ]; then
    VIS=$(PYTHONNOUSERSITE=1 "$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1 || true)
    case "$VIS" in ''|*[!0-9]*) VIS=0 ;; esac
    if [ "$VIS" -ge 4 ]; then NGPU=4; elif [ "$VIS" -ge 2 ]; then NGPU=2; else NGPU=1; fi
  fi
  SEQ_PER_GPU=${SEQ_PER_GPU:-8}
  NUM_WORKER=${NUM_WORKER:-$((2 * NGPU))}
  case "$NGPU" in 1|2|4) ;; *) echo "-g $NGPU: use 1, 2 or 4 (must divide the 892 test sequences)" >&2; exit 2 ;; esac
  case "$SEQ_PER_GPU" in [1-8]) ;; *) echo "SEQ_PER_GPU=$SEQ_PER_GPU: use 1-8" >&2; exit 2 ;; esac
  case "$NUM_WORKER" in ''|*[!0-9]*) echo "NUM_WORKER=$NUM_WORKER is not a number" >&2; exit 2 ;; esac
  for f in test1_right test2_right; do
    [ -s "$DATA_ROOT/$f.h5" ] || { echo "missing $DATA_ROOT/$f.h5 (see README, Datasets)" >&2; exit 1; }
  done
  export HDF5_USE_FILE_LOCKING=FALSE
  export SPCONV_ALGO=Native   # no int32 overflow at large batches; the reported number uses it
  BS=$((SEQ_PER_GPU * NGPU))
  OPTS="$OPTS batch_size=$BS batch_size_val=$BS num_worker=$NUM_WORKER"
  echo "HOI4D evaluation: $NGPU GPU(s) x $SEQ_PER_GPU sequences, $NUM_WORKER loader workers, output in $OUT"
fi

CMD=("$PYTHON" tools/test.py --config-file "configs/$DATASET/$CONFIG.py" --num-gpus "$NGPU" --options $OPTS $EXTRA)
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "cd $REPO && ${CMD[*]}"
  exit 0
fi
check_gpus "$NGPU" ""
cd "$REPO"
ulimit -n 65536 2>/dev/null || true
mkdir -p "$OUT"
"${CMD[@]}"
echo
if [ "$DATASET" = forainetv2 ]; then
  grep -E "Val result|Class_[0-9]+-" "$OUT/train.log" | tail -4 | sed -E 's/^\[[^]]*\] //'
else
  "$PYTHON" scripts/pptr_miou.py "$OUT/train.log" --latest
fi
