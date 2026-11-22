#!/usr/bin/env bash
# Check that the HOI4D test set is in place for scripts/eval_hoi4d.sh (reads only the h5 headers).
#
#   bash scripts/check_hoi4d.sh                    # data/hoi4d
#   DATA_ROOT=/data/hoi4d bash scripts/check_hoi4d.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "${DATA_ROOT:-}" in ""|/*) ;; *) DATA_ROOT=$PWD/$DATA_ROOT ;; esac
ROOT=${DATA_ROOT:-$REPO/data/hoi4d}
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=python; fi
fi
HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PYTHON" - "$ROOT" <<'PY'
import os, sys
import h5py
root = sys.argv[1]
expect = {"test1_right.h5": 500, "test2_right.h5": 392}
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
print("HOI4D test set is complete" if ok else "HOI4D test set is NOT ready (see README, Data)")
sys.exit(0 if ok else 1)
PY
