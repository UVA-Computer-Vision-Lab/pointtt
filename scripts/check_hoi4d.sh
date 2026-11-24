#!/usr/bin/env bash
# Check that the HOI4D files are in place (reads only the h5 headers).
#
#   bash scripts/check_hoi4d.sh                    # the test files (evaluation), in data/hoi4d
#   bash scripts/check_hoi4d.sh --train            # the test files and the training files train2-4.h5
#   DATA_ROOT=/data/hoi4d bash scripts/check_hoi4d.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=0
for a in "$@"; do
  case "$a" in
    --train) TRAIN=1 ;;
    *) echo "unknown argument: $a (use --train to check the training files too)" >&2; exit 2 ;;
  esac
done
case "${DATA_ROOT:-}" in ""|/*) ;; *) DATA_ROOT=$PWD/$DATA_ROOT ;; esac
ROOT=${DATA_ROOT:-$REPO/data/hoi4d}
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=python; fi
fi
HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PYTHON" - "$ROOT" "$TRAIN" <<'PY'
import os, sys
import h5py
root, train = sys.argv[1], sys.argv[2] == "1"
expect = {"test1_right.h5": 500, "test2_right.h5": 392}
if train:
    expect.update({"train2.h5": 750, "train3.h5": 750, "train4.h5": 721})
ok = True
for fn, n in expect.items():
    p = os.path.join(root, fn)
    if not os.path.exists(p):
        print(f"MISSING {p}"); ok = False; continue
    try:
        with h5py.File(p, "r") as f:
            pcd, sem = f["pcd"].shape, f["semantic"].shape
    except (OSError, KeyError, AttributeError) as e:
        print(f"BAD     {fn}: cannot open ({e})"); ok = False; continue
    good = pcd == (n, 300, 8192, 3) and sem == (n, 300, 8192)
    print(f"{'ok     ' if good else 'BAD    '} {fn}: pcd {pcd} semantic {sem}  ({os.path.getsize(p):,} bytes)")
    ok &= good
what = "HOI4D test and training files" if train else "HOI4D test set"
print(f"{what}: complete" if ok else f"{what}: NOT ready (see README, Datasets)")
sys.exit(0 if ok else 1)
PY
