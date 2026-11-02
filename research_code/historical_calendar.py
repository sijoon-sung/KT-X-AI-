"""H-0004 historical-only recorded-transaction mean forecasts; execute on dislab.

This module has no API data loader or API evaluation path. Calendar rates are
learned from the downloaded historical archive, selected on historical February,
saved before the historical March evaluation, and never refitted on that period.
"""
import argparse
import calendar
from datetime import datetime, timedelta, timezone
import hashlib
import io
import csv
import json
from pathlib import Path
import platform
import re
import os
import subprocess
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
HOUR = 3600
ALPHAS = (1, 4, 16, 64)
FLOOR = 1e-6
DT = re.compile(r'^(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$')
FIELDS = ('자전거번호', '대여일시', '대여_대여소ID', '반납일시', '반납_대여소ID')
EVENT_DTYPE = np.dtype([('bike', '<u4'), ('rent', '<i8'), ('origin', '<u4'),
                        ('returned', '<i8'), ('destination', '<u4')])


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return int(parsed.timestamp())


START, TRAIN_END, VALID_END, END = [stamp(x) for x in
    ('2025-08-01', '2026-02-01', '2026-03-01', '2026-04-01')]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def parse_trip(row, positions, month):
    """Normalize source keys/timestamps; return a distinct exclusion reason."""
    values = [row[positions[k]].strip() for k in FIELDS]
    if not all(values):
        return None, 'missing_key'
    bike, rented, origin, returned, destination = values
    times = []
    try:
        for value in (rented, returned):
            match = DT.fullmatch(value)
            if not match:
                return None, 'invalid_datetime'
            normalized = f'{match[1]}T{int(match[2]):02d}:{match[3]}:{match[4] or "00"}'
            times.append(stamp(normalized))
    except ValueError:
        return None, 'invalid_datetime'
    if rented[:7] != month:
        return None, 'filename_month_mismatch'
    if not 0 <= times[1] - times[0] < 86400:
        return None, 'invalid_duration'
    return (bike, times[0], origin, times[1], destination), None


def read_events(archive, catalog):
    station_index, bike_index, parts, covered_months = {}, {}, [], set()
    counters = {k: 0 for k in ('source_rows', 'invalid_width', 'missing_key',
                 'invalid_datetime', 'filename_month_mismatch', 'invalid_duration')}
    expected = {x['name']: x for x in catalog['members']}
    with zipfile.ZipFile(archive) as z:
        files = [x for x in z.infolist() if x.filename.lower().endswith('.csv')]
        if {f.filename for f in files} != set(expected) or len(files) != len(expected):
            raise ValueError('Archive member list differs from the recorded source.')
        for entry in files:
            source = expected[entry.filename]
            h = hashlib.sha256()
            with z.open(entry) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            if entry.file_size != source['bytes'] or h.hexdigest() != source['sha256']:
                raise ValueError('Historical source member bytes differ.')
            rows = []
            with z.open(entry) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding=source['encoding'], errors='strict'))
                header = next(reader)
                if header != source['header']:
                    raise ValueError('Historical source header differs.')
                positions = {k: header.index(k) for k in FIELDS}
                for row in reader:
                    counters['source_rows'] += 1
                    if len(row) != len(header):
                        counters['invalid_width'] += 1
                        continue
                    values, reason = parse_trip(row, positions, source['expected_rental_month'])
                    if reason:
                        counters[reason] += 1
                        continue
                    bike, rented, origin, returned, destination = values
                    rows.append((bike_index.setdefault(bike, len(bike_index)), rented,
                        station_index.setdefault(origin, len(station_index)), returned,
                        station_index.setdefault(destination, len(station_index))))
            if rows:
                covered_months.add(source['expected_rental_month'])
                parts.append(np.array(rows, dtype=EVENT_DTYPE))
            print(json.dumps(dict(stage='source_member', name=entry.filename,
                                  accepted_before_dedup=len(rows))), flush=True)
            del rows
    if not parts:
        raise ValueError('No eligible completed historical trips.')
    events = np.concatenate(parts)
    before = len(events)
    events = np.unique(events)
    counters.update(duplicate_completed_trips=before-len(events), accepted_completed_trips=len(events))
    ids = np.array(list(station_index), dtype=str)
    return events, ids, covered_months, counters


def calendar_axes(times):
    dates = [datetime.fromtimestamp(int(t), KST) for t in times]
    return dict(hour=np.array([d.hour for d in dates]),
        weekday=np.array([d.weekday() for d in dates]),
        day=np.array([(int(t)+32400)//86400 for t in times]),
        interior=np.array([1 < d.day < calendar.monthrange(d.year, d.month)[1] for d in dates]),
        month=np.array([d.strftime('%Y-%m') for d in dates]))


def build_panel(events, ids, covered_months, start=START, end=END):
    times = np.arange(start, end, HOUR, dtype=np.int64)
    counts = np.zeros((len(times), len(ids), 2), dtype=np.int32)
    for kind, time_key, station_key in ((0, 'rent', 'origin'), (1, 'returned', 'destination')):
        indices = (events[time_key]-start)//HOUR
        valid = (indices >= 0) & (indices < len(times))
        np.add.at(counts, (indices[valid], events[station_key][valid], kind), 1)
    axes = calendar_axes(times)
    source_month = np.isin(axes['month'], list(covered_months))
    hours = (axes['hour'] >= 7) & (axes['hour'] <= 21)
    common_recorded = (counts.sum(1) > 0).all(1)
    valid = source_month & axes['interior'] & hours & common_recorded
    # Numeric zero means zero accepted source records only inside declared source scope.
    panel = counts.astype(np.float64)
    panel[~valid] = np.nan
    return dict(y=panel, times=times, ids=ids, valid=valid, **axes), dict(
        candidate_hours=len(times), missing_source_month_hours=int((~source_month).sum()),
        boundary_date_hours=int((source_month & ~axes['interior']).sum()),
        out_of_scope_hours=int((source_month & axes['interior'] & ~hours).sum()),
        unknown_city_record_hours=int((source_month & axes['interior'] & hours & ~common_recorded).sum()),
        eligible_record_hours=int(valid.sum()))


def split_masks(panel):
    times, valid = panel['times'], panel['valid']
    return {name: valid & (times >= start) & (times < end) for name, start, end in
        (('train', START, TRAIN_END), ('validation', TRAIN_END, VALID_END), ('test', VALID_END, END))}


def training_cohort(panel, train):
    y = panel['y'][train]
    totals = y.sum(0)
    day_count = np.zeros(y.shape[1], dtype=int)
    for day in np.unique(panel['day'][train]):
        day_count += (panel['y'][train & (panel['day'] == day)].sum((0, 2)) > 0)
    first = train & (panel['times'] < START+7*86400)
    last = train & (panel['times'] >= TRAIN_END-7*86400)
    early = panel['y'][first].sum((0, 2)) > 0
    late = panel['y'][last].sum((0, 2)) > 0
    return (totals >= 100).all(1) & (day_count >= 60) & early & late


def fit_rates(y, weekday, hour, alpha):
    if not len(y) or not np.isfinite(y).all() or (y < 0).any() or alpha <= 0:
        raise ValueError('Fit requires finite nonnegative historical training counts.')
    baseline = np.maximum((y.sum(0)+.5)/(len(y)+1), FLOOR)
    sums = np.zeros((7, 24, *baseline.shape), dtype=np.float64)
    number = np.zeros((7, 24), dtype=np.int64)
    np.add.at(sums, (weekday, hour), y)
    np.add.at(number, (weekday, hour), 1)
    rates = np.maximum((sums+alpha*baseline)/(number[..., None, None]+alpha), FLOOR)
    return baseline, rates


def deviance(y, prediction):
    if not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('Deviance needs nonnegative finite labels and positive finite means.')
    return 2*(y*np.log(np.where(y > 0, y, 1)/prediction)-y+prediction)


def choose_alpha(train_y, train_weekday, train_hour, valid_y, valid_weekday, valid_hour):
    candidates, fitted = [], []
    for alpha in ALPHAS:
        baseline, rates = fit_rates(train_y, train_weekday, train_hour, alpha)
        candidates.append(dict(alpha=alpha,
            validation_deviance=float(deviance(valid_y, rates[valid_weekday, valid_hour]).mean())))
        fitted.append((baseline, rates))
    selected = min(range(len(candidates)), key=lambda i: candidates[i]['validation_deviance'])
    return candidates[selected]['alpha'], fitted[selected], candidates


def save_model(path, kind, ids, means, alpha):
    metadata = dict(hypothesis_id='H-0004', kind=kind, alpha=alpha, forecast='next full clock hour',
        target='accepted recorded rentals/returns, not inventory or latent demand', hours=[7, 21],
        data_source='historical downloaded transactions only; no API inputs', timezone='Asia/Seoul conditional')
    np.savez_compressed(path, station_ids=ids, means=means, metadata=np.array(json.dumps(metadata)))


def load_prediction(path, station_id, timestamp):
    dt = datetime.fromtimestamp(stamp(timestamp), KST)
    if not 7 <= dt.hour <= 21 or dt.minute or dt.second:
        raise ValueError('Model supports clock hours 07:00 through 21:00 only.')
    with np.load(path, allow_pickle=False) as source:
        matches = np.flatnonzero(source['station_ids'] == station_id)
        if len(matches) != 1:
            raise ValueError('Station is outside the historical training cohort.')
        meta = json.loads(str(source['metadata']))
        means = source['means']
        output = means[dt.weekday(), dt.hour, matches[0]] if meta['kind'] == 'calendar' else means[matches[0]]
    return dict(timestamp=dt.isoformat(), station_id=station_id,
        expected_recorded_rentals=float(output[0]), expected_recorded_returns=float(output[1]),
        target=meta['target'])


def bootstrap(daily, block, repetitions=2000, seed=20261031):
    rng = np.random.default_rng(seed)
    days = len(daily)
    starts = rng.integers(0, days-block+1, (repetitions, (days+block-1)//block))
    indices = (starts[..., None]+np.arange(block)).reshape(repetitions, -1)[:, :days]
    totals = daily[indices].sum(1)
    valid = (totals[:, 0] > 0) & (totals[:, 2] > 0)
    values = 100*(totals[valid, 0]-totals[valid, 1])/totals[valid, 0]
    return dict(block_days=block, requested_replicates=repetitions, valid_replicates=len(values),
                interval=np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None)


def decision(stations, valid_days, test_days, cells, intervals):
    if stations < 500 or valid_days < 14 or test_days < 21 or cells < 100000:
        return 'inconclusive'
    if any(x['valid_replicates'] < 1900 or x['interval'] is None for x in intervals):
        return 'inconclusive'
    if all(x['interval'][0] > 1 for x in intervals):
        return 'supported'
    if all(x['interval'][1] <= 1 for x in intervals):
        return 'not_supported'
    return 'inconclusive'


def fit_main(args):
    started, timer = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'research_code'], cwd=ROOT, check=True)
    record = json.loads((ROOT/'research/hypotheses/H-0004.json').read_text(encoding='utf-8'))
    canonical = json.dumps(record['plan'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert record['status'] in ('preregistered', 'running')
    assert hashlib.sha256(canonical.encode()).hexdigest() == record['preregistration']['plan_sha256']
    input_path = ROOT/'research/datasets/tashu-historical-H0004-v1.json'
    assert digest(input_path) == record['plan']['data']['input_manifest_sha256']
    inputs = json.loads(input_path.read_text(encoding='utf-8'))
    assert digest(args.archive) == inputs['source_archive']['sha256'], 'Not the frozen historical learning archive.'
    catalog_path = ROOT/inputs['source_catalog']
    assert digest(catalog_path) == inputs['source_catalog_sha256']
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    run_dir = ROOT/'research/runs'/args.run_id
    if run_dir.exists():
        raise ValueError('Run directory already exists; preserve it and choose a new run ID.')
    artifacts = run_dir/'artifacts'
    artifacts.mkdir(parents=True)
    events, ids, months, quality = read_events(args.archive, catalog)
    panel, scope_mask = build_panel(events, ids, months)
    del events
    masks = split_masks(panel)
    selected = training_cohort(panel, masks['train'])
    selected_ids = panel['ids'][selected]
    if not len(selected_ids) or not masks['validation'].any() or not masks['test'].any():
        write(run_dir/'attempt.json', dict(status='nonestimable', decision='inconclusive',
            reason='No historical training cohort or internal validation/test observations.',
            code_commit=code_commit, quality=quality, masks=scope_mask))
        raise RuntimeError('Not enough historical data; no fabricated model metric.')
    y = panel['y'][:, selected]
    tr, va, te = [masks[x] for x in ('train', 'validation', 'test')]
    hour, weekday = panel['hour'], panel['weekday']
    print(json.dumps(dict(stage='historical_cohort', stations=len(selected_ids),
          train_hours=int(tr.sum()), validation_hours=int(va.sum()), test_hours=int(te.sum()))), flush=True)
    alpha, (base, rates), choices = choose_alpha(y[tr], weekday[tr], hour[tr], y[va], weekday[va], hour[va])
    save_model(artifacts/'constant.npz', 'constant', selected_ids, base, None)
    save_model(artifacts/'calendar.npz', 'calendar', selected_ids, rates, alpha)
    for filename, expected in (('constant.npz', base), ('calendar.npz', rates)):
        with np.load(artifacts/filename, allow_pickle=False) as model:
            assert np.array_equal(model['means'], expected) and np.array_equal(model['station_ids'], selected_ids)
    selection = dict(selected_alpha=alpha, candidates=choices,
        selected_at_utc=datetime.now(timezone.utc).isoformat(), refitted_after_selection=False,
        model_hashes={f: digest(artifacts/f) for f in ('constant.npz', 'calendar.npz')})
    write(artifacts/'selection.json', selection)
    print(json.dumps(dict(stage='historical_selection_saved', alpha=alpha)), flush=True)
    # Historical test scores are first calculated after coefficients/selection were saved.
    test_y = y[te]
    predictions = dict(constant=np.broadcast_to(base, test_y.shape), calendar=rates[weekday[te], hour[te]])
    losses = {name: deviance(test_y, pred) for name, pred in predictions.items()}
    means = {name: float(value.mean()) for name, value in losses.items()}
    if means['constant'] <= 0:
        raise ValueError('Baseline deviance is zero; relative improvement is undefined.')
    day0 = (VALID_END+32400)//86400
    test_day = panel['day'][te]-day0
    daily = np.zeros((31, 3), dtype=float)
    for day in range(31):
        take = test_day == day
        daily[day] = [losses['constant'][take].sum(), losses['calendar'][take].sum(), int(take.sum())*len(selected_ids)*2]
    intervals = [bootstrap(daily, block) for block in (3, 7)]
    valid_days = len(np.unique(panel['day'][va]))
    test_days = len(np.unique(panel['day'][te]))
    cells = int(test_y.size)
    verdict = decision(len(selected_ids), valid_days, test_days, cells, intervals)
    metric = 100*(means['constant']-means['calendar'])/means['constant']
    secondary = {}
    for k, kind in enumerate(('rentals', 'returns')):
        threshold = max(1., float(np.percentile(y[tr, :, k], 90)))
        peak = test_y[..., k] >= threshold
        secondary[kind] = dict(training_peak_threshold=threshold, peak_test_cells=int(peak.sum()), models={})
        for name, pred in predictions.items():
            absolute = np.abs(test_y[..., k]-pred[..., k])
            secondary[kind]['models'][name] = dict(deviance=float(losses[name][..., k].mean()),
                mae=float(absolute.mean()), peak_mae=float(absolute[peak].mean()) if peak.any() else None)
    np.savez_compressed(artifacts/'prepared_panel.npz', recorded_counts=y, observed_record_hours=panel['valid'],
                        timestamp_unix_seconds=panel['times'], station_ids=selected_ids)
    scope = dict(learning_source='downloaded historical completed trips; no API data',
        source_archive_sha256=inputs['source_archive']['sha256'], source_quality=quality, source_hour_masks=scope_mask,
        source_station_count=len(ids), selected_training_station_count=len(selected_ids),
        excluded_training_station_count=len(ids)-len(selected_ids),
        periods={name: dict(hours=int(mask.sum()), dates=len(np.unique(panel['day'][mask])),
                           cells=int(mask.sum())*len(selected_ids)*2) for name, mask in masks.items()},
        target='accepted recorded transaction counts conditional on source scope, not verified actual demand/stock')
    metrics = dict(calendar_deviance_improvement_percent=metric,
                   constant_mean_poisson_deviance=means['constant'], calendar_mean_poisson_deviance=means['calendar'],
                   eligible_stations=len(selected_ids), evaluation_cells=cells)
    for b in intervals:
        if b['interval'] is not None:
            metrics[f"ci_{b['block_days']}day_lower"] = b['interval'][0]
            metrics[f"ci_{b['block_days']}day_upper"] = b['interval'][1]
    limitations = ['Historical recorded completed-trip counts only; no API training, tuning or evaluation.',
        'Zero count is no accepted record in assumed-covered city hours; complete station recording is not independently proven.',
        'Missing months, month boundaries, nights and unknown city hours are masked; duration>=24h trips are excluded.',
        'Recurring training stations and daytime source scope restrict generalization; operation and station changes remain uncertain.',
        'Naive clocks conditionally interpreted as KST; no independent source-clock verification.',
        'Retrospective historical periods were summarized in legacy work; not untouched independent external test.',
        'Bootstrap is conditional on training cohort and validation-selected alpha; model-selection uncertainty is excluded.',
        'Poisson deviance compares mean forecasts; not Poisson distribution fit, uncertainty calibration, inventory or policy validation.']
    summary = dict(hypothesis_id='H-0004', decision=verdict, metrics=metrics, scope=scope,
        selected_alpha=alpha, validation_candidates=choices, bootstraps=intervals, secondary=secondary,
        daily_test_scores=[dict(day_index=i, constant_deviance_sum=float(row[0]),
                calendar_deviance_sum=float(row[1]), evaluation_cells=int(row[2])) for i, row in enumerate(daily)],
        coefficient_count=dict(constant=int(base.size), calendar=int(rates.size)),
        limitations=limitations, elapsed_seconds=time.perf_counter()-timer,
        selection_saved_before_test=True, api_used=False, reloaded_coefficients_identical=True)
    write(run_dir/'summary.json', summary)
    artifact_records = [dict(id='summary.json', sha256=digest(run_dir/'summary.json'), availability='public aggregate')]
    for filename in ('constant.npz', 'calendar.npz', 'selection.json', 'prepared_panel.npz'):
        artifact_records.append(dict(id=filename, sha256=digest(artifacts/filename), availability='private dislab artifact; not published'))
    manifest = dict(schema_version=1, run_id=args.run_id, hypothesis_id='H-0004',
        plan_sha256=record['preregistration']['plan_sha256'], code_commit=code_commit,
        input_manifest_sha256=digest(input_path), started_at_utc=started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python', 'research_code/historical_calendar.py', 'fit', '--archive', '<PRIVATE_DOWNLOADED_HISTORY>', '--run-id', args.run_id],
        environment=dict(python=platform.python_version(), packages=dict(numpy=np.__version__),
            compute_location='dislab', device='CPU; no GPU training',
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
        seed=20261031, evaluation_scope=scope, metrics=metrics, artifacts=artifact_records, limitations=limitations)
    write(run_dir/'manifest.json', manifest)
    print(json.dumps(dict(stage='completed', decision=verdict, metrics=metrics,
                         seconds=summary['elapsed_seconds'])), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--archive', required=True)
    fit.add_argument('--run-id', required=True)
    predict = sub.add_parser('predict')
    predict.add_argument('--model', required=True)
    predict.add_argument('--station-id', required=True)
    predict.add_argument('--timestamp', required=True)
    args = ap.parse_args()
    if args.mode == 'fit':
        fit_main(args)
    else:
        print(json.dumps(load_prediction(args.model, args.station_id, args.timestamp), ensure_ascii=False))


if __name__ == '__main__':
    main()
