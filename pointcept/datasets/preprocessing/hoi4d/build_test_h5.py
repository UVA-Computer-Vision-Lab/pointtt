"""Split the HOI4D semantic segmentation test set (test_wolabel_seg.h5 + label.npy, from the
"Sem Seg Test Set" of https://hoi4d.github.io/) into the two-file layout expected by
pointcept/datasets/hoi4d.py (`test1_right.h5` for seqs 0..499, `test2_right.h5` for seqs 500..891).

    python pointcept/datasets/preprocessing/hoi4d/build_test_h5.py
    (defaults: --test_h5 data/hoi4d/SemSegChallenge/test_wolabel_seg.h5
               --label_npy data/hoi4d/SemSegChallenge/label.npy  --out_dir data/hoi4d)

Copies one sequence at a time (about 60 MB), so it needs well under 1 GB of RAM.
Each file is written as `<name>.tmp` and renamed when it is complete.
"""
import argparse
import os

import h5py
import numpy as np


def write_rows(f, key, like, lo, hi, row):
    """Dataset `key` = rows lo..hi-1 of an array shaped like `like`, one row (sequence) at a time."""
    dset = f.create_dataset(key, shape=(hi - lo,) + like.shape[1:], dtype=like.dtype, compression=None)
    for i in range(lo, hi):
        dset[i - lo] = row(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_h5', default='data/hoi4d/SemSegChallenge/test_wolabel_seg.h5')
    ap.add_argument('--label_npy', default='data/hoi4d/SemSegChallenge/label.npy')
    ap.add_argument('--out_dir', default='data/hoi4d')
    args = ap.parse_args()

    print('reading', args.test_h5)
    with h5py.File(args.test_h5, 'r') as src:
        pcd = src['pcd']                  # (892, 300, 8192, 3)
        center = src['center']            # (892, 300, 3)
        print('  pcd', pcd.shape, pcd.dtype)
        print('  center', center.shape, center.dtype)

        print('reading', args.label_npy)
        label = np.load(args.label_npy, mmap_mode='r')    # (892, 300, 8192)
        print('  label', label.shape, label.dtype)
        assert pcd.shape[0] == label.shape[0] == 892

        def label_row(i):
            # a fresh mapping per row: a long-lived one keeps every page it has touched in memory
            return np.array(np.load(args.label_npy, mmap_mode='r')[i])

        splits = [(0, 500, 'test1_right.h5'), (500, 892, 'test2_right.h5')]
        os.makedirs(args.out_dir, exist_ok=True)
        for lo, hi, name in splits:
            out_path = os.path.join(args.out_dir, name)
            tmp_path = out_path + '.tmp'
            print(f'writing {out_path}: seqs {lo}..{hi}')
            try:
                # one dataset after the other, each filled before the next is created: this is the
                # allocation order of create_dataset(data=...), hence the same bytes on disk
                with h5py.File(tmp_path, 'w') as f:
                    write_rows(f, 'pcd', pcd, lo, hi, lambda i: pcd[i])
                    write_rows(f, 'center', center, lo, hi, lambda i: center[i])
                    write_rows(f, 'semantic', label, lo, hi, label_row)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            os.replace(tmp_path, out_path)
    print('done')


if __name__ == '__main__':
    main()
