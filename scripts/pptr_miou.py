#!/usr/bin/env python3
"""Compute the HOI4D pptr_mIoU from the evaluation log (train.log) of tools/test.py.

Pointcept's `SemSegEvaluator` reports a flat `np.mean(iou_class)` over all
classes (including ones that have no ground truth — those just show up as
`iou=0`). PPTr's HOI4D evaluation skips:
    1. classes in `skiplist = [0, 1]`  (void / sky placeholders)
    2. classes with no GT (`total_class[c] < 0.1`)

The log doesn't print raw `total_class[c]`, but a class with no GT will
always have `iou=0.0000 AND accuracy=0.0000`. We use that as the proxy; on the
HOI4D test set it keeps exactly the 39 classes that have ground truth.

Usage (scripts/test.sh runs it):
    python scripts/pptr_miou.py path/to/train.log
"""
from __future__ import annotations

import argparse
import re
import sys

SKIPLIST = {0, 1}

# `Val result: mIoU/mAcc/allAcc 0.3093/0.4197/0.7563.`
HEAD_RE = re.compile(
    r"Val result:\s*mIoU/mAcc/allAcc\s+(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"
)
# `Class_2-2 Result: iou/accuracy 0.5807/0.7292`
CLS_RE = re.compile(
    r"Class_(\d+)-\d+\s+Result:\s+iou/accuracy\s+([\d.]+)/([\d.]+)"
)
# Try to also capture the timestamp / epoch context that precedes a block.
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def parse_blocks(text: str):
    """Yield (timestamp, raw_miou, raw_macc, raw_allacc, [(cls, iou, acc), ...]).

    Each "Val result:" line starts a new block; we collect all subsequent
    `Class_N-N Result:` lines until the next "Val result:" or end-of-file.
    """
    # Find positions of every Val result header.
    headers = [m for m in HEAD_RE.finditer(text)]
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]

        raw = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        cls_lines = [
            (int(c.group(1)), float(c.group(2)), float(c.group(3)))
            for c in CLS_RE.finditer(chunk)
        ]
        # Try to grab a timestamp from the line containing "Val result"
        line_start = text.rfind("\n", 0, start) + 1
        line = text[line_start:start]
        ts_match = TS_RE.match(line)
        ts = ts_match.group(1) if ts_match else "n/a"

        yield ts, raw, cls_lines


def pptr_miou(classes):
    """Compute PPTr's mIoU = mean over classes not in skiplist and with GT."""
    active = [
        (c, iou) for c, iou, acc in classes
        if c not in SKIPLIST and (iou > 0 or acc > 0)
    ]
    if not active:
        return 0.0, []
    return sum(iou for _, iou in active) / len(active), active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to the evaluation log (train.log)")
    args = ap.parse_args()

    with open(args.log) as f:
        text = f.read()

    blocks = list(parse_blocks(text))
    if not blocks:
        print(f"no Val result blocks found in {args.log}", file=sys.stderr)
        sys.exit(1)

    # the most recent block, in a single line
    ts, (raw_miou, raw_macc, raw_allacc), classes = blocks[-1]
    miou_p, active = pptr_miou(classes)
    print(
        f"[pptr_score] block={len(blocks)} ts={ts} "
        f"raw_mIoU={raw_miou:.4f} pptr_mIoU={miou_p:.4f} "
        f"active_classes={len(active)} raw_mAcc={raw_macc:.4f} "
        f"raw_allAcc={raw_allacc:.4f}"
    )
    if len(active) != 39:
        print(f"WARNING: {len(active)} active classes, the HOI4D test labels have 39; "
              "a class without any correct prediction may have been dropped", file=sys.stderr)


if __name__ == "__main__":
    main()
