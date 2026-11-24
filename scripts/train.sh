#!/usr/bin/env bash
# Train with a config of configs/DATASET/ (tools/train.py), Pointcept style: the code is copied to
# EXP/code and runs from there; checkpoints go to EXP/model, the log to EXP/train.log.
#
#   bash scripts/train.sh -d forainetv2 -c pointttt-forainetv2-stage1 -g 1 -n exp/forainetv2/stage1
#   bash scripts/train.sh -d forainetv2 -c pointttt-forainetv2-stage2 -g 1 -n exp/forainetv2/stage2 \
#       -w exp/forainetv2/stage1/model/model_best.pth
#   bash scripts/train.sh -n exp/forainetv2/stage2 -r true      # continue from EXP/model/model_last.pth
#
# Options:
#   -d DATASET  config directory: forainetv2 or hoi4d
#   -c CONFIG   config name in configs/DATASET/ (without .py)
#   -g GPUS     number of GPUs (default 1); batch_size and num_worker in the configs are totals
#   -n EXP      experiment directory (default: exp/DATASET/CONFIG)
#   -w FILE     initial weights (stage 2: the stage-1 checkpoint)
#   -r true     resume EXP: its config.py, code/ and model/model_last.pth (-d, -c, -w not needed)
#   -e NAME     experiment name (only used by the optional W&B logging, enable_wandb=True)
#   -p PYTHON   interpreter of the environment (default: ./env/bin/python if it exists, else python)
#   -o "K=V .." extra config overrides, e.g. -o "epoch=2 eval_epoch=2"
# Environment variables (all optional): PYTHON; DATA_ROOT (data directory, default data/DATASET);
# PTV3_LN_SPLIT_AFFINE (default 1: a LayerNorm variant for training that is about 1.2x faster and
# equal up to rounding; the released checkpoints were trained with it); FORCE=1 starts afresh in an
# EXP that already holds checkpoints; DRY_RUN=1 prints the command instead of running it.
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

DATASET="" CONFIG="" NGPU=1 EXP="" WEIGHT="" RESUME=false NAME="" EXTRA=""
while getopts "d:c:g:n:w:r:e:p:o:h" opt; do
  case $opt in
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    g) NGPU=$OPTARG ;;
    n) EXP=$OPTARG ;;
    w) WEIGHT=$OPTARG ;;
    r) RESUME=$OPTARG ;;
    e) NAME=$OPTARG ;;
    p) PYTHON=$OPTARG ;;
    o) EXTRA=$OPTARG ;;
    h) usage 0 ;;
    *) usage ;;
  esac
done
[ "$OPTIND" -gt $# ] || { echo "unexpected argument: ${!OPTIND}" >&2; usage; }
case "$RESUME" in
  true|True|1|yes) RESUME=True ;;
  false|False|0|no|"") RESUME=False ;;
  *) echo "-r $RESUME: use true or false" >&2; exit 2 ;;
esac
case "$NGPU" in ''|*[!0-9]*|0) echo "-g $NGPU is not a positive number" >&2; exit 2 ;; esac
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=python; fi
fi

if [ "$RESUME" = True ]; then
  [ -n "$EXP" ] || { echo "-r true needs -n EXP (the directory of the run to continue)" >&2; exit 2; }
  EXP=$(abspath "$EXP")
  CFG_FILE=$EXP/config.py
  WEIGHT=$EXP/model/model_last.pth
  [ -s "$CFG_FILE" ] || { echo "nothing to resume: $CFG_FILE not found" >&2; exit 1; }
  [ -s "$WEIGHT" ] || { echo "nothing to resume: $WEIGHT not found (no epoch finished yet?)" >&2; exit 1; }
  [ -f "$EXP/code/tools/train.py" ] || { echo "nothing to resume: $EXP/code/ not found" >&2; exit 1; }
else
  [ -n "$DATASET" ] && [ -n "$CONFIG" ] || { echo "-d DATASET and -c CONFIG are required" >&2; usage; }
  [ -f "$REPO/configs/$DATASET/$CONFIG.py" ] || { echo "no such config: configs/$DATASET/$CONFIG.py" >&2; exit 2; }
  EXP=$(abspath "${EXP:-$REPO/exp/$DATASET/$CONFIG}")
  if [ -n "$WEIGHT" ]; then
    WEIGHT=$(abspath "$WEIGHT")
    [ -s "$WEIGHT" ] || { echo "initial weights not found: $WEIGHT" >&2; exit 1; }
  fi
  if ls "$EXP"/model/*.pth > /dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; then
    echo "$EXP already holds checkpoints: continue it with -r true, choose another -n, or set FORCE=1" >&2
    exit 1
  fi
fi
NAME=${NAME:-$(basename "$EXP")}
# the run reads its config from its own code snapshot (EXP/code/configs/...)
[ "$RESUME" = True ] || CFG_FILE=$EXP/code/configs/$DATASET/$CONFIG.py

# the --options parser splits values at commas and drops blanks
for p in "$REPO" "$EXP" ${WEIGHT:+"$WEIGHT"} ${DATA_ROOT:+"$(abspath "$DATA_ROOT")"}; do
  case "$p" in
    *[[:space:]]*|*,*|*\[*|*\]*|*\(*|*\)*|*\'*|*\"*)
      echo "paths with blanks, commas, brackets or quotes are not supported: $p" >&2; exit 2 ;;
  esac
done

OPTS="save_path=$EXP experiment_name=$NAME"
[ "$RESUME" = True ] && OPTS="$OPTS resume=True"
[ -n "$WEIGHT" ] && OPTS="$OPTS weight=$WEIGHT"
if [ -n "${DATA_ROOT:-}" ]; then
  D=$(abspath "$DATA_ROOT")
  OPTS="$OPTS data.train.data_root=$D data.val.data_root=$D"
fi
CMD=("$PYTHON" "$EXP/code/tools/train.py" --config-file "$CFG_FILE" --num-gpus "$NGPU" --options $OPTS $EXTRA)
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "cd $REPO && PYTHONPATH=$EXP/code ${CMD[*]}"
  exit 0
fi
check_gpus "$NGPU" "; use a smaller -g (batch_size and num_worker in the configs are totals over all GPUs;
for HOI4D stage 1 see the 1-GPU replay command in configs/hoi4d/pointttt-hoi4d-stage1.py)"

if [ "$RESUME" != True ]; then
  mkdir -p "$EXP/model"
  rm -rf "$EXP/code"
  mkdir -p "$EXP/code"
  # snapshot of the code this run uses (a later resume runs the same code)
  (cd "$REPO" && tar --exclude=__pycache__ --exclude='*.pyc' -cf - pointcept tools scripts configs) | tar -xf - -C "$EXP/code"
fi
echo "experiment: $EXP"
echo "config:     $CFG_FILE"
echo "GPUs:       $NGPU"
echo "weights:    ${WEIGHT:-none}"

cd "$REPO"
export PYTHONPATH=$EXP/code
export PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE
export PTV3_LN_SPLIT_AFFINE=${PTV3_LN_SPLIT_AFFINE:-1}
ulimit -n 65536 2>/dev/null || true
exec "${CMD[@]}"
