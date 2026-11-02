"""H-0003 development-only stock-structure diagnostic; run on dislab."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (10, 30, 60)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort(stock, times, actual, start, end):
    origins = np.flatnonzero((times >= start + 3600) & (times < end - 3600)
                             & ((times + 32400) % 3600 == 0))
    offsets = np.array(HORIZONS) // 10
    target_indices = origins[None, :] + offsets[:, None]
    if not np.all(times[target_indices] == times[origins][None, :] + np.array(HORIZONS)[:, None] * 60):
        raise ValueError('Observation grid is not contiguous.')
    current, future = stock[origins], stock[target_indices]
    current_actual, future_actual = actual[origins], actual[target_indices]
    for values, stamps in ((current, current_actual), (future, future_actual)):
        finite = np.isfinite(values)
        if np.any((values[finite] < 0) | (values[finite] != np.floor(values[finite]))) or np.any(stamps[finite] < 0):
            raise ValueError('Input values or request timestamps violate the audited contract.')
    elapsed = future_actual - current_actual[None, :, :]
    valid = (np.isfinite(current) & np.isfinite(future).all(0)
             & (np.abs(elapsed - np.array(HORIZONS)[:, None, None] * 60) <= 120).all(0))
    days = ((times[origins] + 32400) // 86400 - (start + 32400) // 86400).astype(int)
    return dict(origins=origins, current=current, future=future, valid=valid, days=days)


def sufficient(data, days=28):
    n_stations = data['current'].shape[1]
    n = np.zeros((days, n_stations, 9), dtype=np.float64)
    sums = np.zeros((3, days, n_stations, 9), dtype=np.float64)
    squares = np.zeros_like(sums)
    chosen = data['valid'] & (data['current'] >= 2) & (data['current'] <= 10)
    row, station = np.nonzero(chosen)
    state = data['current'][row, station].astype(int) - 2
    day = data['days'][row]
    np.add.at(n, (day, station, state), 1)
    for h in range(3):
        y = data['future'][h, row, station].astype(np.float64)
        np.add.at(sums[h], (day, station, state), y)
        np.add.at(squares[h], (day, station, state), y * y)
    return n, sums, squares


def station_ratios(n, sums, squares, eligible):
    usable = eligible & (n >= 2)
    square_mean = np.divide(sums * sums, n, out=np.zeros_like(sums), where=n > 0)
    variance = np.divide(squares - square_mean, n - 1,
                         out=np.zeros_like(sums), where=usable)
    variance = np.maximum(variance, 0.)  # Roundoff guard; never trim a real positive variance.
    numerator = np.where(usable, n * variance, 0).sum(-1)
    denominator = np.where(usable, sums, 0).sum(-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def block_weights(days, block, repetitions, seed=20261031):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days - block + 1, (repetitions, (days + block - 1) // block))
    indices = (starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, :days]
    return np.stack([(indices == d).sum(1) for d in range(days)], axis=1).astype(float)


def resample(n, sums, squares, eligible, station_set, block, repetitions=2000):
    weights = block_weights(len(n), block, repetitions)
    values, coverage = [], []
    flat = [x.reshape(len(n), -1) for x in (n, sums, squares)]
    for start in range(0, repetitions, 64):
        reweighted = [(weights[start:start + 64] @ x).reshape(-1, *eligible.shape) for x in flat]
        ratios = station_ratios(*reweighted, eligible)[:, station_set]
        finite_share = np.isfinite(ratios).mean(1)
        valid = finite_share >= .95
        if np.any(valid):
            values.extend(np.nanmedian(ratios[valid], axis=1).tolist())
        coverage.extend(finite_share.tolist())
    return dict(block_days=block, valid_replicates=len(values), requested_replicates=repetitions,
                minimum_station_coverage=float(min(coverage)) if coverage else None,
                interval=np.percentile(values, [2.5, 97.5]).tolist() if values else None)


def decide(stations, pairs, bootstraps):
    if stations < 100 or pairs < 10000 or any(b['valid_replicates'] < 1900 or b['interval'] is None for b in bootstraps):
        return 'inconclusive'
    if all(b['interval'][1] < .8 for b in bootstraps):
        return 'supported'
    if all(b['interval'][0] >= .8 for b in bootstraps):
        return 'not_supported'
    return 'inconclusive'


def distribution(values):
    values = values[np.isfinite(values)]
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                mean=float(np.mean(values)) if len(values) else None,
                quantiles_10_25_50_75_90=np.percentile(values, [10, 25, 50, 75, 90]).tolist() if len(values) else [],
                share_below_1=float(np.mean(values < 1)) if len(values) else None)


def transitions(data):
    current, future, valid = data['current'], data['future'], data['valid']
    buckets = [('0', 0, 0), ('1', 1, 1), ('2', 2, 2), ('3', 3, 3), ('4', 4, 4), ('5', 5, 5),
               ('6..10', 6, 10), ('11..20', 11, 20), ('21+', 21, np.inf)]
    result = []
    for label, lo, hi in buckets:
        chosen = valid & (current >= lo) & (current <= hi)
        count = int(chosen.sum())
        if not count:
            continue
        baseline = current[chosen]
        for h, minutes in enumerate(HORIZONS):
            y = future[h][chosen]
            change = np.abs(y - baseline)
            result.append(dict(current_stock=label, horizon_minutes=minutes, pairs=count,
                zero_share=float(np.mean(y == 0)), unchanged_share=float(np.mean(y == baseline)),
                mean_absolute_change=float(np.mean(change)), p95_absolute_change=float(np.percentile(change, 95))))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    started, timer = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'research_code'], cwd=ROOT, check=True)
    record = json.loads((ROOT / 'research/hypotheses/H-0003.json').read_text(encoding='utf-8'))
    canonical = json.dumps(record['plan'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert record['status'] in ('preregistered', 'running')
    assert record['preregistration']['plan_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
    inputs_path = ROOT / 'research/datasets/tashu-primary-v1.json'
    inputs = json.loads(inputs_path.read_text(encoding='utf-8'))
    assert digest(inputs_path) == record['plan']['data']['input_manifest_sha256']
    assert digest(args.panel) == inputs['derived_files'][0]['sha256']
    with np.load(args.panel, allow_pickle=False) as source:
        stock, times = source['stock'], source['scheduled_at_unix_seconds']
        actual, ids = source['observed_at_unix_seconds'], source['station_ids']
    period = record['plan']['evaluation']['periods'][0]
    start, end = [int(datetime.fromisoformat(period[k]).timestamp()) for k in ('start', 'end')]
    data = cohort(stock, times, actual, start, end)
    n, sums, squares = sufficient(data)
    count, observed_days = n.sum(0), (n > 0).sum(0)
    eligible = (count >= 30) & (observed_days >= 14)
    all_ratios = np.stack([station_ratios(count, sums[h].sum(0), squares[h].sum(0), eligible) for h in range(3)])
    station_set = np.isfinite(all_ratios[1])
    station_count = int(station_set.sum())
    eligible_pairs = int(np.where(eligible, count, 0).sum())
    run_dir = ROOT / 'research/runs' / args.run_id
    if not station_count:
        write(run_dir / 'attempt.json', dict(status='nonestimable', decision='inconclusive',
             reason='No finite primary station ratios; no fabricated numeric metric.', code_commit=code_commit))
        raise RuntimeError('Primary ratio is not estimable; see attempt.json.')
    print(json.dumps(dict(stage='cohort', stations=station_count, eligible_pairs=eligible_pairs,
                         origins=len(data['origins']), common_pairs=int(data['valid'].sum()))), flush=True)
    bootstraps = []
    for block in (3, 7):
        bootstraps.append(resample(n, sums[1], squares[1], eligible, station_set, block))
        print(json.dumps(dict(stage='bootstrap', **bootstraps[-1])), flush=True)
    verdict = decide(station_count, eligible_pairs, bootstraps)
    strict = (count >= 60) & (observed_days >= 14)
    sensitivity = np.stack([station_ratios(count, sums[h].sum(0), squares[h].sum(0), strict) for h in range(3)])
    metrics = dict(median_station_conditional_fano_30min=float(np.median(all_ratios[1, station_set])),
                   eligible_stations=station_count, eligible_pairs=eligible_pairs,
                   eligible_strata=int(eligible.sum()), common_pairs=int(data['valid'].sum()))
    for b in bootstraps:
        metrics[f"bootstrap_{b['block_days']}day_valid"] = b['valid_replicates']
        if b['interval'] is not None:
            metrics[f"ci_{b['block_days']}day_lower"] = b['interval'][0]
            metrics[f"ci_{b['block_days']}day_upper"] = b['interval'][1]
    limitations = [
        'Development-period diagnostic, already used by H-0001/H-0002; not independent forecast validation.',
        'Primary population: observed current stock 2..10 with sufficiently repeated station-state strata.',
        'Coarse conditioning omits time and other history; finite dependent samples and complete-case selection limit inference.',
        'Block intervals are diagnostic, not a formal goodness-of-fit rejection of every Poisson process or annual behavior.',
        'Observation likelihood and rental/return event intensities are different quantities; operator actions are unobserved.',
        'No GCN, flow matching, CMP or other final architecture is selected by this diagnostic.'
    ]
    summary = dict(hypothesis_id='H-0003', decision=verdict, metrics=metrics, bootstraps=bootstraps,
        scope=dict(period=period, candidate_station_origins=int(data['valid'].size),
                   common_complete_pairs=int(data['valid'].sum()), station_ids=len(ids),
                   eligible_stations=station_count, eligible_strata=int(eligible.sum()),
                   undefined_mean_station_count=int(np.any(eligible, axis=1).sum())-station_count,
                   daily_common_pairs=[dict(day_index=d, pairs=int(data['valid'][data['days'] == d].sum())) for d in range(28)]),
        horizon_diagnostics=[dict(horizon_minutes=h, n30=distribution(all_ratios[k]),
                                  n60_sensitivity=distribution(sensitivity[k])) for k, h in enumerate(HORIZONS)],
        transitions=transitions(data), elapsed_seconds=time.perf_counter()-timer, limitations=limitations)
    private = run_dir / 'artifacts'
    private.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private / 'station_statistics.npz', station_ids=ids, day_counts=n,
                        day_sums=sums, day_squares=squares, eligible=eligible, ratios=all_ratios)
    write(run_dir / 'summary.json', summary)
    manifest = dict(schema_version=1, run_id=args.run_id, hypothesis_id='H-0003',
        plan_sha256=record['preregistration']['plan_sha256'], code_commit=code_commit,
        input_manifest_sha256=digest(inputs_path), started_at_utc=started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python', 'research_code/stock_dispersion.py', '--panel', '<PRIVATE_AUDITED_PANEL>', '--run-id', args.run_id],
        environment=dict(python=platform.python_version(), packages=dict(numpy=np.__version__),
            compute_location='dislab, user-authorized remote Windows workstation', device='CPU; no GPU training',
            thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
        seed=20261031, evaluation_scope=summary['scope'], metrics=metrics,
        artifacts=[dict(id='summary.json', sha256=digest(run_dir / 'summary.json'), availability='public aggregate'),
                   dict(id='station-level sufficient statistics', sha256=digest(private / 'station_statistics.npz'),
                        availability='private dislab filesystem; not published')], limitations=limitations)
    write(run_dir / 'manifest.json', manifest)
    print(json.dumps(dict(decision=verdict, metrics=metrics, seconds=summary['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
