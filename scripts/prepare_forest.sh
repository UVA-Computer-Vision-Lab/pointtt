#!/usr/bin/env bash
# Download FOR-instanceV2 from Zenodo (record 16742708) and lay it out the way
# pointcept/datasets/forainetv2.py expects:
#
#   data/forainetv2/test_data/*.ply              test_data.zip: 377 MB, 1.4 GB unpacked
#   data/forainetv2/train_val_data/*.ply         train_val_data.zip: 2.4 GB, 11 GB unpacked (only with --train)
#   data/forainetv2/meta_data/{train,val,test}_list.txt   the official split lists (47 / 16 / 28 plots)
#
#   bash scripts/prepare_forest.sh                              # test split only (evaluation), into data/forainetv2
#   bash scripts/prepare_forest.sh --train                      # also the training and validation plots
#   DATA_ROOT=/data/forainetv2 bash scripts/prepare_forest.sh    # elsewhere; then pass the same DATA_ROOT to
#                                                                # scripts/test.sh and scripts/train.sh
#                                                                # (or link it to data/forainetv2)
#
# The three BlueCat_RN_merged_trees_* plots of the archives are in none of the lists (neither trained on
# nor evaluated).
# Needs curl, md5sum and unzip (or python). Can be run again at any time: finished steps are skipped, an
# interrupted download continues from its .part file, an interrupted extraction starts over. The zip is not
# needed once it is unpacked. Do not start it twice at the same time on the same DATA_ROOT.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPLITS="test"
for a in "$@"; do
  case "$a" in
    --train) SPLITS="train val test" ;;
    *) echo "unknown argument: $a (use --train to get the training and validation plots too)" >&2; exit 2 ;;
  esac
done
case "${DATA_ROOT:-}" in ""|/*) ;; *) DATA_ROOT=$PWD/$DATA_ROOT ;; esac
ROOT=${DATA_ROOT:-$REPO/data/forainetv2}
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/env/bin/python" ]; then PYTHON=$REPO/env/bin/python; else PYTHON=$(command -v python3 || command -v python || true); fi
fi
[ -n "$PYTHON" ] || { echo "no python found: set PYTHON=/path/to/python" >&2; exit 1; }
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd -P)

# unzip is not installed everywhere: settle how to unpack before anything is downloaded
if command -v unzip > /dev/null; then
  unzip_to() { unzip -q "$1" -d "$2"; }
else
  "$PYTHON" -c "import zipfile, zlib" || {
    echo "no unzip, and $PYTHON cannot replace it: install unzip" >&2; exit 1; }
  unzip_to() {
    echo "  (no unzip on this machine, using $PYTHON -m zipfile)"
    "$PYTHON" -m zipfile -e "$1" "$2"
  }
fi

URL=https://zenodo.org/records/16742708/files

md5_ok() {  # md5_ok <file> <md5>
  [ "$(md5sum < "$1" | cut -d ' ' -f 1)" = "$2" ]
}

# fetch <file> <md5>: afterwards $ROOT/<file> exists and has this md5. The download goes to <file>.part
# and is resumable; a part is deleted only when it is complete and still has the wrong md5.
fetch() {
  local f=$ROOT/$1 rc code resumed
  command -v md5sum > /dev/null || { echo "md5sum not found" >&2; exit 1; }
  if [ -f "$f" ]; then
    if md5_ok "$f" "$2"; then echo "$1: md5 ok"; return 0; fi
    if [ -L "$f" ]; then
      echo "$f is a link to a file with the wrong md5 (expected $2): remove the link" >&2
      exit 1
    fi
    if [ -e "$f.part" ]; then
      echo "$f has the wrong md5 and $f.part exists as well: delete one of them" >&2
      exit 1
    fi
    echo "$1: wrong md5, treating it as an unfinished download"
    mv "$f" "$f.part"
  fi
  while true; do
    # a finished part must not go to curl again: the server answers a resume at the end with HTTP 416
    if [ -f "$f.part" ] && md5_ok "$f.part" "$2"; then break; fi
    resumed=no
    if [ -s "$f.part" ]; then resumed=yes; echo "$1: resuming $1.part"; else echo "$1: downloading"; fi
    command -v curl > /dev/null || { echo "curl not found" >&2; exit 1; }
    rc=0
    code=$(curl -fL --retry 5 -C - -o "$f.part" -w '%{http_code}' "$URL/$1?download=1") || rc=$?
    if [ "$rc" -ne 0 ] && [ "$code" != 416 ]; then
      echo "$1: download failed (curl exit $rc, HTTP $code)" >&2
      if [ -s "$f.part" ]; then
        echo "  $f.part is kept: run this script again to resume, or delete it to start from scratch" >&2
      fi
      exit 1
    fi
    if md5_ok "$f.part" "$2"; then break; fi
    rm -f "$f.part"
    if [ "$resumed" = no ]; then
      echo "$1: downloaded completely, but its md5 is not $2; deleted it" >&2
      exit 1
    fi
    echo "$1: the resumed download is complete (HTTP $code) but has the wrong md5;"
    echo "  deleted $1.part, starting from scratch"
  done
  mv "$f.part" "$f"
  echo "$1: md5 ok"
}

# get <name> <md5 of <name>.zip>: $ROOT/<name>/, unpacked from the zip (which holds <name>/*.ply).
# The directory is moved into place in one step, so one that exists is complete and is left alone.
get() {
  local tmp=$ROOT/$1.unpacking
  if [ -d "$ROOT/$1" ]; then return 0; fi
  if [ -e "$ROOT/$1" ] || [ -L "$ROOT/$1" ]; then
    echo "$ROOT/$1 exists but is not a directory (a dangling link?): remove it and run this script again" >&2
    exit 1
  fi
  fetch "$1.zip" "$2"
  if [ -e "$tmp" ]; then
    echo "$1.zip: removing what an interrupted run left in $1.unpacking"
    rm -rf "$tmp"
  fi
  echo "$1.zip: unpacking"
  mkdir "$tmp"
  unzip_to "$ROOT/$1.zip" "$tmp"
  mv "$tmp/$1" "$ROOT/$1"
  rm -rf "$tmp"
}
case " $SPLITS " in *" train "*) get train_val_data 5a63cc1cbe88edd9ebec28ad7e46f79b ;; esac
get test_data 1c00a0f0b89f03b74064432162619136

mkdir -p "$ROOT/meta_data"
for split in $SPLITS; do
  list=$REPO/pointcept/datasets/preprocessing/forainetv2/meta_data/${split}_list.txt
  cmp -s "$list" "$ROOT/meta_data/${split}_list.txt" || cp "$list" "$ROOT/meta_data/"
done

# every listed plot must exist and be complete
"$PYTHON" - "$ROOT" $SPLITS <<'PY'
import os, sys
root, splits = sys.argv[1], sys.argv[2:]
BYTES = dict(char=1, uchar=1, int8=1, uint8=1, short=2, ushort=2, int16=2, uint16=2, int=4, uint=4,
             int32=4, uint32=4, float=4, float32=4, double=8, float64=8)

def truncated(path):
    """True if a binary PLY is shorter than its header says. Reads the header only."""
    with open(path, "rb") as f:
        head = f.read(65536)
    end = head.find(b"end_header")
    size = head.find(b"\n", end) + 1                   # header length
    if not head.startswith(b"ply") or end < 0 or size == 0:
        return True
    try:
        count, binary = 0, False
        for line in head[:end].decode("ascii", "replace").splitlines():
            w = line.split()
            if w[:1] == ["format"]:
                binary = w[1].startswith("binary")
            elif w[:1] == ["element"]:
                count = int(w[2])
            elif w[:1] == ["property"]:
                if w[1] not in BYTES:                  # list property: the header does not fix the size
                    return False
                size += count * BYTES[w[1]]
    except (IndexError, ValueError):
        return True
    return binary and os.path.getsize(path) < size

ok, redo = True, set()
for split in splits:
    sub, expect = {"train": ("train_val_data", 47), "val": ("train_val_data", 16), "test": ("test_data", 28)}[split]
    names = [l.strip() for l in open(os.path.join(root, "meta_data", f"{split}_list.txt")) if l.strip()]
    files = [os.path.join(sub, n + ".ply") for n in names]
    missing = [p for p in files if not os.path.exists(os.path.join(root, p))]
    short = [p for p in files if p not in missing and truncated(os.path.join(root, p))]
    print(f"{split:5s}: {len(names)} plots listed (expected {expect}), {len(missing)} missing, {len(short)} truncated")
    for p in (missing + short)[:5]:
        print("   missing or truncated:", p)
    ok &= len(names) == expect and not missing and not short
    if missing or short:
        redo.add(sub)
for sub in sorted(redo):
    print(f"{sub} is incomplete: delete {os.path.join(root, sub)} and run this script again")
sys.exit(0 if ok else 1)
PY
echo "FOR-instanceV2 ($SPLITS) ready at $ROOT"
