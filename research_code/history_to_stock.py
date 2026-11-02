"""H-0005: frozen historical flow priors -> official September stock evaluation.

No fitting, API calibration, model selection, capacity fitting or optimizer.
Current API stock is an initial condition; future API stock is a score label only.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
START = int(datetime(2026, 9, 1, tzinfo=KST).timestamp())
END = int(datetime(2026, 10, 1, tzinfo=KST).timestamp())
BOOT_START = int(datetime(2026, 9, 17, tzinfo=KST).timestamp())
DELTA_HOURS = .5


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8', newline='\n')


def numeric_contract(rate_sums, duration=.5):
    rate_sums = np.asarray(rate_sums, dtype=float)
    if not np.isfinite(rate_sums).all() or (rate_sums < 0).any() or not np.isfinite(duration) or duration < 0:
        raise ValueError('Rates/duration must be finite and nonnegative.')
    maximum = float(rate_sums.max(initial=0)*duration)
    cutoff = max(16, int(np.ceil(poisson.isf(1e-12, maximum)))+1)
    if cutoff > 512:
        raise ValueError('Preregistered numerical iteration limit exceeded.')
    tail = float(poisson.sf(cutoff, maximum))
    mean_bound = float(maximum*poisson.sf(cutoff-1, maximum)+tail*maximum/(1-tail))
    if tail > 1e-10 or mean_bound > 1e-8:
        raise ValueError('Preregistered probability/mean tail error limit exceeded.')
    return dict(cutoff=cutoff, maximum_expected_attempts=maximum,
                tail_probability_bound=tail, mean_error_bound=mean_bound)


def transition_tables(arrival, departure, cutoff, duration=.5, chunk=512):
    """Normalized truncated uniformization; no physical upper capacity.

    Backward recursion computes E[g(X(t))|X(0)=b] for g=I[b=0] and g=b.
    The right boundary for g=b has a known linear expression because more
    than cutoff starting bikes cannot hit zero in <=cutoff embedded jumps.
    """
    a, d = np.asarray(arrival, float), np.asarray(departure, float)
    if a.shape != d.shape or not np.isfinite(a).all() or not np.isfinite(d).all() or (a < 0).any() or (d < 0).any():
        raise ValueError('Incompatible or invalid historical rates.')
    shape = a.shape
    af, df = a.ravel(), d.ravel()
    pzero = np.empty((len(af), cutoff+1))
    means = np.empty_like(pzero)
    max_b = 2*cutoff
    for start in range(0, len(af), chunk):
        aa, dd = af[start:start+chunk], df[start:start+chunk]
        q = aa+dd
        up = np.divide(aa, q, out=np.zeros_like(q), where=q > 0)
        down = np.divide(dd, q, out=np.zeros_like(q), where=q > 0)
        mass = np.exp(-q*duration)
        total = mass.copy()
        f = np.zeros((len(q), max_b+1)); f[:, 0] = 1
        g = np.broadcast_to(np.arange(max_b+1, dtype=float), f.shape).copy()
        sum_zero, sum_mean = mass[:, None]*f[:, :cutoff+1], mass[:, None]*g[:, :cutoff+1]
        for n in range(1, cutoff+1):
            f_lower = np.concatenate((f[:, :1], f[:, :-1]), axis=1)
            f_upper = np.concatenate((f[:, 1:], np.zeros((len(q), 1))), axis=1)
            g_lower = np.concatenate((g[:, :1], g[:, :-1]), axis=1)
            right = max_b+1+(n-1)*(up-down)
            g_upper = np.concatenate((g[:, 1:], right[:, None]), axis=1)
            f = up[:, None]*f_upper+down[:, None]*f_lower
            g = up[:, None]*g_upper+down[:, None]*g_lower
            mass = mass*(q*duration)/n
            total += mass
            sum_zero += mass[:, None]*f[:, :cutoff+1]
            sum_mean += mass[:, None]*g[:, :cutoff+1]
        pzero[start:start+len(q)] = sum_zero/total[:, None]
        means[start:start+len(q)] = sum_mean/total[:, None]
    return pzero.reshape(*shape, cutoff+1), means.reshape(*shape, cutoff+1)


def large_state_mean(stock, arrival, departure, cutoff, duration=.5):
    q = arrival+departure
    m = q*duration
    embedded_mean = m*poisson.cdf(cutoff-1, m)/poisson.cdf(cutoff, m)
    difference = np.divide(arrival-departure, q, out=np.zeros_like(q), where=q > 0)
    return stock+difference*embedded_mean


def pair_data(stock, times, actual, api_ids, trained_ids):
    if len(set(api_ids.tolist())) != len(api_ids) or len(set(trained_ids.tolist())) != len(trained_ids):
        raise ValueError('Duplicate station identifiers are not allowed.')
    by_id = {key: i for i, key in enumerate(api_ids)}
    model_cols = np.array([i for i, key in enumerate(trained_ids) if key in by_id], dtype=int)
    api_cols = np.array([by_id[trained_ids[i]] for i in model_cols], dtype=int)
    local_hour = ((times+32400)//3600)%24
    origin = np.flatnonzero((times >= START) & (times+1800 < END)
        & ((times+32400)%3600 == 0) & (local_hour >= 7) & (local_hour <= 21))
    target = origin+3
    if (target >= len(times)).any() or not np.all(times[target]-times[origin] == 1800):
        raise ValueError('September target observation grid is damaged.')
    current = stock[origin][:, api_cols]
    future = stock[target][:, api_cols]
    now_stamp, target_stamp = actual[origin][:, api_cols], actual[target][:, api_cols]
    for values, stamps in ((current, now_stamp), (future, target_stamp)):
        finite = np.isfinite(values)
        if (values[finite] < 0).any() or (values[finite] != np.floor(values[finite])).any() or (stamps[finite] < 0).any():
            raise ValueError('Selected September values violate the audited input contract.')
    finite = np.isfinite(current) & np.isfinite(future)
    delay_now, delay_target = now_stamp-times[origin, None], target_stamp-times[target, None]
    elapsed = target_stamp-now_stamp
    timing = (delay_now >= 0) & (delay_now <= 120) & (delay_target >= 0) & (delay_target <= 120) & (np.abs(elapsed-1800) <= 120)
    valid = finite & timing
    day = (times[origin]+32400)//86400
    return dict(current=current, future=future, valid=valid, elapsed=elapsed,
        times=times[origin], weekday=(day+3)%7, hour=local_hour[origin],
        day_index=((times[origin]-BOOT_START)//86400).astype(int), model_cols=model_cols,
        scope=dict(trained_ids=len(trained_ids), api_ids=len(api_ids), matched_ids=len(model_cols),
            unmatched_trained_ids=len(trained_ids)-len(model_cols), unmatched_api_ids=len(api_ids)-len(model_cols),
            candidate_origins=len(origin), candidate_cells=int(current.size),
            missing_current_or_target=int((~finite).sum()), invalid_timing_with_observed_values=int((finite & ~timing).sum()),
            valid_pairs=int(valid.sum()), period_start='2026-09-01T00:00:00+09:00',
            period_end_exclusive='2026-10-01T00:00:00+09:00', source_dates='2026-09-17..2026-09-30'))


def forecast(current, weekday, hour, model_cols, rates, tables, cutoff, kind):
    b = np.where(np.isfinite(current), current, 0).astype(np.int64)
    if (b < 0).any():
        raise ValueError('Initial stock must be nonnegative.')
    index = np.minimum(b, cutoff)
    if kind == 'calendar':
        key = (weekday[:, None], (hour-7)[:, None], model_cols[None, :], index)
        p0, mean = tables[0][key], tables[1][key]
        selected_rates = rates[weekday[:, None], hour[:, None], model_cols[None, :]]
    else:
        key = (model_cols[None, :], index)
        p0, mean = tables[0][key], tables[1][key]
        selected_rates = np.broadcast_to(rates[model_cols], (*current.shape, 2))
    high = b > cutoff
    if high.any():
        # Historical model output order: rentals, returns.
        high_mean = large_state_mean(b, selected_rates[..., 1], selected_rates[..., 0], cutoff)
        p0[high], mean[high] = 0., high_mean[high]
    if not np.isfinite(p0).all() or not np.isfinite(mean).all() or (p0 < -1e-12).any() or (p0 > 1+1e-12).any() or (mean < -1e-10).any():
        raise ValueError('Numerical transition output outside its probability/count domain.')
    return np.clip(p0, 0, 1), np.maximum(mean, 0)


def bootstrap(daily, block, repetitions=2000, seed=20261031):
    rng = np.random.default_rng(seed)
    days = len(daily)
    starts = rng.integers(0, days-block+1, (repetitions, (days+block-1)//block))
    indices = (starts[..., None]+np.arange(block)).reshape(repetitions, -1)[:, :days]
    totals = daily[indices].sum(1)
    valid = totals[:, 2] > 0
    reduction = (totals[valid, 0]-totals[valid, 1])/totals[valid, 2]
    return dict(block_days=block, requested_replicates=repetitions, valid_replicates=len(reduction),
                interval=np.percentile(reduction, [2.5, 97.5]).tolist() if len(reduction) else None)


def decision(ids, pairs, dates, point, intervals):
    if ids < 500 or pairs < 10000 or dates < 10 or any(b['valid_replicates'] < 1900 or b['interval'] is None for b in intervals):
        return 'inconclusive'
    if point > .005 and all(b['interval'][0] > .005 for b in intervals):
        return 'supported'
    if all(b['interval'][1] <= .005 for b in intervals):
        return 'not_supported'
    return 'inconclusive'


def group_scores(data, predictions):
    result, bins = {}, {}
    y0 = data['future'] == 0
    for label, mask in [('positive', data['valid'] & (data['current'] > 0)),
                        ('zero', data['valid'] & (data['current'] == 0)), ('all', data['valid'])]:
        result[label] = dict(pairs=int(mask.sum()), observed_zero_share=float(y0[mask].mean()) if mask.any() else None, models={})
        for name, (p, mean) in predictions.items():
            result[label]['models'][name] = dict(brier=float(((p[mask]-y0[mask])**2).mean()) if mask.any() else None,
                count_mean_mae=float(np.abs(mean[mask]-data['future'][mask]).mean()) if mask.any() else None,
                predicted_zero_mean=float(p[mask].mean()) if mask.any() else None)
    for name, (p, _) in predictions.items():
        values, truth = p[data['valid']], y0[data['valid']]
        indices = np.minimum((values*10).astype(int), 9)
        bins[name] = [dict(bin=i, lower=i/10, upper=(i+1)/10, count=int((indices == i).sum()),
            predicted_mean=float(values[indices == i].mean()) if (indices == i).any() else None,
            observed_mean=float(truth[indices == i].mean()) if (indices == i).any() else None) for i in range(10)]
    return result, bins


def run(args):
    started, timer = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'research_code'], cwd=ROOT, check=True)
    record = json.loads((ROOT/'research/hypotheses/H-0005.json').read_text(encoding='utf-8'))
    canonical = json.dumps(record['plan'], ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert record['status'] in ('preregistered', 'running')
    assert hashlib.sha256(canonical.encode()).hexdigest() == record['preregistration']['plan_sha256']
    input_path = ROOT/'research/datasets/tashu-history-to-stock-H0005-v1.json'
    assert digest(input_path) == record['plan']['data']['input_manifest_sha256']
    inputs = json.loads(input_path.read_text(encoding='utf-8'))
    learning, evaluation = inputs['learning'], inputs['evaluation']
    assert digest(ROOT/learning['source_run_manifest']) == learning['source_run_manifest_sha256']
    assert digest(ROOT/evaluation['api_manifest']) == evaluation['api_manifest_sha256']
    directory = ROOT/'research/runs'/args.run_id
    if directory.exists():
        raise ValueError('Run directory exists; preserve it and choose a new ID.')
    artifacts = directory/'artifacts'; artifacts.mkdir(parents=True)
    rates, models, hashes, ids = {}, {}, {}, None
    for entry in learning['model_artifacts']:
        path = Path(args.rate_model_dir)/entry['id']
        assert digest(path) == entry['sha256']
        kind = entry['id'].removesuffix('.npz')
        with np.load(path, allow_pickle=False) as source:
            if ids is not None and not np.array_equal(ids, source['station_ids']):
                raise ValueError('Historical model station order differs.')
            ids, rates[kind] = source['station_ids'].copy(), source['means'].copy()
        hashes[kind] = entry['sha256']
    try:
        numeric = numeric_contract(np.concatenate([rates['calendar'].sum(-1).ravel(), rates['constant'].sum(-1).ravel()]))
    except ValueError as error:
        write(directory/'attempt.json', dict(status='numerical_gate_failed', decision='inconclusive',
            reason=str(error), code_commit=code, api_opened=False))
        raise
    cutoff = numeric['cutoff']
    # All transition tables depend solely on frozen historical weights, before the API panel is opened.
    for kind in ('constant', 'calendar'):
        selected = rates[kind] if kind == 'constant' else rates[kind][:, 7:22]
        models[kind] = transition_tables(selected[..., 1], selected[..., 0], cutoff)
        np.savez_compressed(artifacts/f'kernel_{kind}.npz', pzero=models[kind][0], means=models[kind][1])
    kernel_finished = datetime.now(timezone.utc).isoformat()
    print(json.dumps(dict(stage='historical_tables_before_api', **numeric)), flush=True)
    assert digest(args.panel) == evaluation['audited_panel']['sha256']
    with np.load(args.panel, allow_pickle=False) as source:
        data = pair_data(source['stock'], source['scheduled_at_unix_seconds'], source['observed_at_unix_seconds'], source['station_ids'], ids)
    predictions = {'persistence': ((data['current'] == 0).astype(float), data['current'].copy())}
    for kind in ('constant', 'calendar'):
        predictions[kind] = forecast(data['current'], data['weekday'], data['hour'], data['model_cols'], rates[kind], models[kind], cutoff, kind)
    positive = data['valid'] & (data['current'] > 0)
    if not positive.any():
        write(directory/'attempt.json', dict(status='nonestimable', decision='inconclusive', reason='No September positive-stock evaluation pairs.', code_commit=code, scope=data['scope']))
        raise ValueError('Primary Brier reduction is not estimable.')
    if ((data['day_index'] < 0) | (data['day_index'] >= 14)).any():
        raise ValueError('Unexpected September source dates outside the frozen bootstrap calendar.')
    y0 = data['future'] == 0
    loss_base = (predictions['persistence'][0]-y0)**2
    loss_calendar = (predictions['calendar'][0]-y0)**2
    point = float((loss_base[positive]-loss_calendar[positive]).mean())
    daily = np.zeros((14, 3))
    for day in range(14):
        mask = positive & (data['day_index'][:, None] == day)
        daily[day] = [loss_base[mask].sum(), loss_calendar[mask].sum(), int(mask.sum())]
    intervals = [bootstrap(daily, block) for block in (3, 7)]
    dates = int((daily[:, 2] > 0).sum())
    verdict = decision(data['scope']['matched_ids'], int(positive.sum()), dates, point, intervals)
    groups, bins = group_scores(data, predictions)
    for entry in learning['model_artifacts']:
        assert digest(Path(args.rate_model_dir)/entry['id']) == entry['sha256'], 'Frozen learning weights changed.'
    np.savez_compressed(artifacts/'predictions.npz', times=data['times'], model_cols=data['model_cols'],
        valid=data['valid'], initial=data['current'], target=data['future'],
        calendar_zero=predictions['calendar'][0], calendar_mean=predictions['calendar'][1],
        constant_zero=predictions['constant'][0], constant_mean=predictions['constant'][1])
    metrics = dict(positive_brier_reduction=point, positive_brier_persistence=groups['positive']['models']['persistence']['brier'],
        positive_brier_calendar=groups['positive']['models']['calendar']['brier'],
        positive_pairs=int(positive.sum()), matched_training_ids=data['scope']['matched_ids'], positive_dates=dates)
    for item in intervals:
        if item['interval'] is not None:
            metrics[f"ci_{item['block_days']}day_lower"], metrics[f"ci_{item['block_days']}day_upper"] = item['interval']
    limitations = ['Official September API evaluation, previously exposed in legacy experiments; not an untouched independent test.',
        'Frozen historical completed-flow priors are effective rates, not identified latent demand or empty/non-full exposure-corrected intensities.',
        'Independent memoryless attempted events, no physical upper capacity, operator rebalancing or state-correction component.',
        'Past-to-September drift, source recording assumptions, station-ID changes and unmatched stations limit transfer.',
        'Current API stock is an evaluation initial condition; request clocks are not server update times and KST is conditional.',
        'Daytime exact-ID complete pairs over about two weeks; block intervals exclude rate-estimation uncertainty.',
        'Primary current-positive empty-risk score does not establish all-stock, joint dependence, operational usefulness or final GCN/FM architecture.']
    summary = dict(hypothesis_id='H-0005', decision=verdict, metrics=metrics, scope=data['scope'],
        groups=groups, reliability_bins=bins, numerical_contract=numeric, bootstraps=intervals,
        daily_positive_scores=[dict(day_index=i, persistence_loss=float(row[0]), calendar_loss=float(row[1]), pairs=int(row[2])) for i, row in enumerate(daily)],
        actual_elapsed_seconds=dict(min=int(data['elapsed'][data['valid']].min()), max=int(data['elapsed'][data['valid']].max())),
        current_zero_recovery_share=1-groups['zero']['observed_zero_share'] if groups['zero']['pairs'] else None,
        learned_from_api=False, api_parameter_selection=False, official_evaluation_month='2026-09',
        kernel_finished_before_api_open=kernel_finished, frozen_rate_hashes_unchanged=True,
        elapsed_seconds=time.perf_counter()-timer, limitations=limitations)
    write(directory/'summary.json', summary)
    artifact_records = [dict(id='summary.json', sha256=digest(directory/'summary.json'), availability='public aggregate')]
    artifact_records += [dict(id=x.name, sha256=digest(x), availability='private dislab transition/prediction artifact; not published') for x in artifacts.glob('*.npz')]
    manifest = dict(schema_version=1, run_id=args.run_id, hypothesis_id='H-0005',
        plan_sha256=record['preregistration']['plan_sha256'], code_commit=code, input_manifest_sha256=digest(input_path),
        started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python','research_code/history_to_stock.py','--panel','<PRIVATE_AUDITED_API_PANEL>',
                 '--rate-model-dir','<PRIVATE_FROZEN_HISTORY_MODELS>','--run-id',args.run_id],
        environment=dict(python=platform.python_version(),packages=dict(numpy=np.__version__,scipy=scipy.__version__),
            compute_location='dislab',device='CPU; no API training or fitting',
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
        seed=20261031,evaluation_scope=data['scope'],metrics=metrics,artifacts=artifact_records,limitations=limitations)
    write(directory/'manifest.json', manifest)
    print(json.dumps(dict(stage='completed',decision=verdict,metrics=metrics,groups=groups,seconds=summary['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',required=True)
    ap.add_argument('--rate-model-dir',required=True)
    ap.add_argument('--run-id',required=True)
    run(ap.parse_args())
