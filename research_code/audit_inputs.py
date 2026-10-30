"""Rebuild archived primary API observations; emit only a public hash manifest."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    prepared = Path(args.prepared)
    source_files, metadata_ids, raw_ids = [], set(), set()
    with np.load(prepared / 'data/observed_panel.npz', allow_pickle=False) as old:
        ids = old['station_ids'].astype(str)
        times = old['scheduled_at_unix_seconds']
        original = old['stock']
        original_actual = old['observed_at_unix_seconds']
    rebuilt = np.full(original.shape, np.nan, dtype=np.float32)
    actual = np.full(original.shape, -1, dtype=np.int64)
    id_index = pd.Index(ids)
    counts = dict(raw_rows=0, invalid_rows=0, duplicate_keys=0,
                  off_schedule_rows=0, extra_poll_rows=0)
    min_raw, max_raw = None, None
    with zipfile.ZipFile(args.archive) as archive:
        names = sorted(n for n in archive.namelist() if re.fullmatch(
            r'tashu/(data/stock_\d{8}\.csv\.gz|meta/stations_\d{8}\.json\.gz)', n))
        assert len(names) == 88, 'Archive membership changed; inspect before registration.'
        for name in names:
            raw_path = prepared / 'raw' / Path(name).relative_to('tashu')
            content = archive.read(name)
            assert raw_path.read_bytes() == content, 'Raw copy differs from archive.'
            source_files.append(dict(id=name, bytes=len(content),
                                     sha256=hashlib.sha256(content).hexdigest()))
            if '/meta/' in name:
                meta = json.loads(gzip.decompress(content))
                metadata_ids.update(str(x['id']) for x in meta['results'])
                continue
            frame = pd.read_csv(raw_path, dtype={'id': 'string'})
            assert set(frame.columns) == {'ts', 'id', 'parking_count'}
            counts['raw_rows'] += len(frame)
            timestamp = pd.to_datetime(frame.ts, errors='coerce')
            value = pd.to_numeric(frame.parking_count, errors='coerce')
            raw_ids.update(frame.id.dropna().astype(str))
            ok = (timestamp.notna() & frame.id.notna() & np.isfinite(value)
                  & value.ge(0) & value.eq(np.floor(value)))
            counts['invalid_rows'] += int((~ok).sum())
            # These inputs were preliminarily clean; fail on any new ambiguity.
            assert ok.all(), 'Invalid raw row requires an explicit revised audit rule.'
            stamp = timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64) - 32400
            min_raw = int(stamp.min()) if min_raw is None else min(min_raw, int(stamp.min()))
            max_raw = int(stamp.max()) if max_raw is None else max(max_raw, int(stamp.max()))
            frame = frame.assign(actual=stamp, value=value.to_numpy())
            duplicate = frame.duplicated(['actual', 'id'], keep=False)
            counts['duplicate_keys'] += int(duplicate.sum())
            assert not duplicate.any(), 'Duplicate keys need explicit reconciliation.'
            slot = stamp // 600 * 600
            frame = frame.assign(slot=slot)
            on_schedule = stamp - slot <= 120
            counts['off_schedule_rows'] += int((~on_schedule).sum())
            frame = frame.loc[on_schedule].copy()
            first = frame.groupby('slot').actual.transform('min')
            keep = frame.actual.eq(first)
            counts['extra_poll_rows'] += int((~keep).sum())
            frame = frame.loc[keep]
            rows = (frame.slot.to_numpy() - times[0]) // 600
            cols = id_index.get_indexer(frame.id.astype(str))
            assert (cols >= 0).all()
            assert np.isnan(rebuilt[rows, cols]).all(), 'Cross-file key overlap.'
            rebuilt[rows, cols] = frame.value.to_numpy(dtype=np.float32)
            actual[rows, cols] = frame.actual.to_numpy(dtype=np.int64)
    assert np.array_equal(np.array(sorted(metadata_ids | raw_ids)), ids)
    assert times[0] == min_raw // 600 * 600 and times[-1] == max_raw // 600 * 600
    assert np.array_equal(times, np.arange(times[0], times[-1] + 600, 600))
    assert np.array_equal(rebuilt, original, equal_nan=True), 'Panel values differ.'
    assert np.array_equal(actual, original_actual), 'Request timestamps differ.'
    present = np.isfinite(rebuilt)
    manifest = dict(
        schema_version=1, dataset_id='tashu-primary-stock-v1-20260917-20261030',
        source='User-provided archived primary Tashu API stock and daily metadata',
        public_raw_data=False, files=source_files,
        derived_files=[dict(id='observed_panel.npz', sha256=sha(prepared / 'data/observed_panel.npz')),
                       dict(id='stations.csv', sha256=sha(prepared / 'data/stations.csv'))],
        audit_code_sha256=sha(__file__),
        audit=dict(**counts, station_ids=len(ids), scheduled_slots=len(times),
                   retained_cells=int(present.sum()),
                   all_station_missing_slots=int((~present.any(1)).sum()),
                   reproduced_panel_exactly=True, reproduced_request_timestamps_exactly=True),
        observation_contract=dict(
            timezone='Asia/Seoul conditional interpretation; OS timezone not independently proven',
            timestamp='Collector request clock; not the server inventory update time',
            grid_seconds=600, accepted_request_delay_seconds=[0, 120],
            multiple_polls='earliest valid request per slot', missing='NaN, never filled',
            scope='API-reported parking_count; not transactions, latent demand or verified usability'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'audit': manifest['audit'], 'manifest_sha256': sha(out)}))


if __name__ == '__main__':
    main()
