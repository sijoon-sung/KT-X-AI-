"""Execute preregistered H-0001, retaining the calendar during block resampling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def day_shifts(panel):
    """panel[day, 144 slots, station]; missing station-days have no estimate."""
    eligible = np.isfinite(panel).all(axis=1)
    fine = (panel == 0).mean(axis=1)
    difference = np.mean([100 * np.abs((panel[:, p::3, :] == 0).mean(1) - fine)
                          for p in range(3)], axis=0)
    return np.where(eligible, difference, np.nan), eligible


def bootstrap(daily, block, repetitions=2000):
    rng = np.random.default_rng(20261031)
    length = len(daily)
    starts = rng.integers(0, length - block + 1,
                          size=(repetitions, (length + block - 1) // block))
    indices = (starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, :length]
    sampled = daily[indices]
    if np.any(~np.isfinite(sampled).any(1)):
        return None
    return np.percentile(np.nanmean(sampled, axis=1), [2.5, 97.5])


def decision(intervals, days):
    if days < 28 or any(x is None for x in intervals):
        return 'inconclusive'
    if all(x[0] > 5 for x in intervals):
        return 'supported'
    if all(x[1] <= 5 for x in intervals):
        return 'not_supported'
    return 'inconclusive'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    record = json.loads((ROOT / 'research/hypotheses/H-0001.json').read_text(encoding='utf-8'))
    plan = record['plan']
    frozen = record['preregistration']
    canonical = json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert frozen['plan_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
    assert record['status'] in ('preregistered', 'running')
    input_path = ROOT / 'research/datasets/tashu-primary-v1.json'
    assert digest(input_path) == plan['data']['input_manifest_sha256']
    inputs = json.loads(input_path.read_text(encoding='utf-8'))
    assert digest(args.panel) == inputs['derived_files'][0]['sha256']
    with np.load(args.panel, allow_pickle=False) as data:
        stock, times = data['stock'], data['scheduled_at_unix_seconds']
    start = int(datetime.fromisoformat('2026-09-18T00:00:00+09:00').timestamp())
    indices = np.searchsorted(times, np.arange(start, start + 42 * 86400, 600))
    assert np.array_equal(times[indices], np.arange(start, start + 42 * 86400, 600))
    shifts, eligible = day_shifts(stock[indices].reshape(42, 144, -1))
    count = eligible.sum(1)
    daily = np.divide(np.nansum(shifts, axis=1), count,
                      out=np.full(42, np.nan), where=count > 0)
    intervals = [bootstrap(daily, block) for block in (3, 7)]
    days = int(np.isfinite(daily).sum())
    verdict = decision(intervals, days)
    metrics = {'mean_daily_abs_zero_share_shift_pp': float(np.nanmean(daily)),
               'eligible_dates': days, 'eligible_station_days': int(eligible.sum()),
               'candidate_station_days': int(eligible.size)}
    for block, interval in zip((3, 7), intervals):
        if interval is not None:
            metrics[f'ci_{block}day_lower_pp'] = float(interval[0])
            metrics[f'ci_{block}day_upper_pp'] = float(interval[1])
    limitations = [
        'Asia/Seoul is a conditional interpretation; source OS timezone is not independently proven.',
        'Complete-case station-days are selectively represented; missing calendar dates remain missing.',
        'Ten-minute polls are a reference, not continuous-time truth, user failures or unmet demand.',
        'Short one-season period; block bootstrap is descriptive and cannot establish annual generalization.'
    ]
    run_dir = ROOT / 'research/runs' / args.run_id
    summary = dict(hypothesis_id='H-0001', decision=verdict, metrics=metrics,
                   daily=[dict(date=str(np.datetime64('2026-09-18') + i),
                               eligible_stations=int(count[i]),
                               abs_shift_pp=float(x) if np.isfinite(x) else None)
                          for i, x in enumerate(daily)], limitations=limitations)
    write(run_dir / 'summary.json', summary)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(schema_version=1, run_id=args.run_id, hypothesis_id='H-0001',
                    plan_sha256=frozen['plan_sha256'], code_commit=commit,
                    input_manifest_sha256=digest(input_path), started_at_utc=started,
                    finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    command=['python', 'research_code/observation_resolution.py', '--panel',
                             '<PRIVATE_OBSERVED_PANEL>', '--run-id', args.run_id],
                    environment=dict(python=platform.python_version(), packages=dict(numpy=np.__version__),
                                     rng='numpy.random.default_rng / PCG64'), seed=20261031,
                    evaluation_scope=f'Paired 42 calendar dates; {days} eligible dates; complete station-days only; no model training.',
                    metrics=metrics,
                    artifacts=[dict(id='summary.json', sha256=digest(run_dir / 'summary.json'),
                                    availability='public repository aggregate')], limitations=limitations)
    write(run_dir / 'manifest.json', manifest)
    print(json.dumps(dict(decision=verdict, metrics=metrics)))


if __name__ == '__main__':
    main()
