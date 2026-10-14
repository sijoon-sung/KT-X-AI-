"""Build an OD graph only from trips before a training cutoff."""
import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np

from preprocess import norm_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--before', default='2026-02-01')
    ap.add_argument('--out', default='data/processed/od_train_before_2026-02-01.npy')
    args = ap.parse_args()
    with np.load(args.data) as f:
        stations = f['stations'].astype(str)
    lookup = {sid: i for i, sid in enumerate(stations)}
    od = np.zeros((len(stations), len(stations)), dtype=np.int32)
    included = 0
    with zipfile.ZipFile(args.zip) as zf:
        for name in sorted(n for n in zf.namelist() if n.lower().endswith('.csv')):
            ym = re.search(r'(\d\d)년(\d\d)월', name)
            prefix = f'20{ym[1]}-{ym[2]}' if ym else None
            if prefix and prefix > args.before[:7]:
                continue
            count = 0
            with zf.open(name) as fh:
                rows = csv.reader(io.TextIOWrapper(fh, encoding='cp949', errors='replace'))
                next(rows)
                for row in rows:
                    if len(row) < 19:
                        continue
                    rent_time = norm_dt(row[1])
                    if (rent_time is None or rent_time >= args.before or
                            (prefix and not rent_time.startswith(prefix))):
                        continue
                    a, b = lookup.get(row[2]), lookup.get(row[10])
                    if a is None or b is None:
                        continue
                    od[a, b] += 1
                    count += 1
            included += count
            print(f'{name}: {count:,}', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, od)
    print(f'saved {args.out}; trips={included:,}; edges={(od > 0).sum():,}', flush=True)


if __name__ == '__main__':
    main()
