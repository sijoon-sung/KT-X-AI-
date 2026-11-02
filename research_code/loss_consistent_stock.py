"""H-0006: same frozen stock distribution, loss-consistent point summaries.

No learning, refitting, probability calibration, API model selection or search.
The absolute-loss action is fixed at the lower median by decision theory.
"""
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
import scipy

import history_to_stock as h

ROOT = Path(__file__).resolve().parents[1]


def state_pmf(initial, arrival, departure, cutoff, duration=.5):
    """Forward uniformization on b+[-K,K], reflecting only at physical zero.

    Conditional on at most K attempted jumps, no other boundary is reachable
    before the final recursion. Negative grid states keep exactly zero mass.
    This is the same normalized truncated Poisson mixture as H-0005.
    """
    b, a, d = (np.asarray(x, dtype=float) for x in (initial, arrival, departure))
    if b.ndim != 1 or b.shape != a.shape or b.shape != d.shape:
        raise ValueError('Expected equal one-dimensional state/rate arrays.')
    if not all(np.isfinite(x).all() for x in (b, a, d)) or (b < 0).any() or (b != np.floor(b)).any() or (a < 0).any() or (d < 0).any():
        raise ValueError('States must be integer nonnegative and rates finite nonnegative.')
    if cutoff < 1 or not np.isfinite(duration) or duration < 0:
        raise ValueError('Invalid numerical contract.')
    q = a+d
    up = np.divide(a, q, out=np.zeros_like(q), where=q > 0)
    down = np.divide(d, q, out=np.zeros_like(q), where=q > 0)
    f = np.zeros((len(b), 2*cutoff+1)); f[:, cutoff] = 1
    mass = np.exp(-q*duration); total = mass.copy()
    probabilities = mass[:, None]*f
    zero_index = cutoff-b.astype(np.int64)
    rows = np.flatnonzero((zero_index >= 0) & (zero_index <= 2*cutoff))
    zero = zero_index[rows]
    for n in range(1, cutoff+1):
        next_f = np.zeros_like(f)
        next_f[:, 1:] += up[:, None]*f[:, :-1]
        # A departure attempt at zero stays at zero, never at a negative grid cell.
        zero_mass = f[rows, zero].copy()
        f[rows, zero] = 0
        next_f[:, :-1] += down[:, None]*f[:, 1:]
        next_f[rows, zero] += down[rows]*zero_mass
        f = next_f
        mass = mass*(q*duration)/n
        total += mass
        probabilities += mass[:, None]*f
    probabilities /= total[:, None]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or np.max(np.abs(probabilities.sum(1)-1), initial=0) > 1e-10:
        raise ValueError('Forward probability mass numerical gate failed.')
    grid = b[:, None]+np.arange(-cutoff, cutoff+1)
    if (probabilities[grid < 0] != 0).any():
        raise ValueError('Probability leaked below physical zero.')
    return grid, probabilities


def summaries(grid, probabilities):
    cdf = np.cumsum(probabilities, axis=1)
    mean = (probabilities*grid).sum(1)
    quantiles = [grid[np.arange(len(grid)), np.argmax(cdf >= p, axis=1)] for p in (.05, .5, .95)]
    lower, median, upper = quantiles
    pzero = (probabilities*(grid == 0)).sum(1)
    mean_loss = (probabilities*np.abs(grid-mean[:, None])).sum(1)
    median_loss = (probabilities*np.abs(grid-median[:, None])).sum(1)
    advantage = mean_loss-median_loss
    if (advantage < -1e-10).any():
        raise ValueError('Median Bayes-action numerical optimality gate failed.')
    return dict(mean=mean, median=median, lower=lower, upper=upper, pzero=pzero,
                expected_mae_advantage=advantage)


def integer_crps(grid, probabilities, labels):
    """CDF squared-error sum, including labels outside finite numerical support."""
    labels = np.asarray(labels, dtype=float)
    cdf = np.cumsum(probabilities, axis=1)
    inside = ((cdf-(grid >= labels[:, None]))**2).sum(1)
    return inside+np.maximum(grid[:, 0]-labels, 0)+np.maximum(labels-grid[:, -1]-1, 0)


def group_scores(data, predictions, crps):
    b, truth, valid = data['current'], data['future'], data['valid']
    yzero = truth == 0
    result = {}
    groups = [('all', valid), ('zero', valid & (b == 0)), ('positive', valid & (b > 0)),
        ('stock_1', valid & (b == 1)), ('stock_2_5', valid & (b >= 2) & (b <= 5)),
        ('stock_6_20', valid & (b >= 6) & (b <= 20)), ('stock_21_plus', valid & (b >= 21))]
    for name, mask in groups:
        n = int(mask.sum())
        if not n:
            result[name] = dict(pairs=0, point_outputs={}, distribution=None)
            continue
        points = {}
        for kind, pred in [('persistence', b), ('mean', predictions['mean']), ('median', predictions['median'])]:
            error = pred[mask].astype(float)-truth[mask]
            points[kind] = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())), signed_bias=float(error.mean()))
        distribution = dict(crps=float(crps[mask].mean()), persistence_crps=points['persistence']['mae'],
            interval_90_coverage=float(((truth[mask] >= predictions['lower'][mask]) & (truth[mask] <= predictions['upper'][mask])).mean()),
            interval_90_mean_width=float((predictions['upper'][mask]-predictions['lower'][mask]).mean()),
            brier=float(((predictions['pzero'][mask]-yzero[mask])**2).mean()),
            observed_zero_share=float(yzero[mask].mean()), predicted_zero_mean=float(predictions['pzero'][mask].mean()))
        result[name] = dict(pairs=n,point_outputs=points,distribution=distribution,
            median_equals_current_share=float((predictions['median'][mask] == b[mask]).mean()),
            median_zero_share=float((predictions['median'][mask] == 0).mean()),
            mean_expected_mae_advantage=float(predictions['expected_mae_advantage'][mask].mean()))
    return result


def classify(ids, pairs, dates, point, intervals):
    if ids < 500 or pairs < 100000 or dates < 10 or any(x['valid_replicates'] < 1900 or x['interval'] is None for x in intervals):
        return 'inconclusive'
    if point > .01 and all(x['interval'][0] > .01 for x in intervals):
        return 'supported'
    if all(x['interval'][1] <= .01 for x in intervals):
        return 'not_supported'
    return 'inconclusive'


def run(args):
    timer, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    record = json.loads((ROOT/'research/hypotheses/H-0006.json').read_text(encoding='utf-8'))
    assert record['status'] in ('preregistered','running')
    canonical = json.dumps(record['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == record['preregistration']['plan_sha256']
    input_path = ROOT/'research/datasets/tashu-loss-consistent-stock-H0006-v1.json'
    assert h.digest(input_path) == record['plan']['data']['input_manifest_sha256']
    inputs = json.loads(input_path.read_text(encoding='utf-8'))
    assert h.digest(ROOT/inputs['parent_manifest']) == inputs['parent_manifest_sha256']
    learning, evaluation = inputs['learning'], inputs['evaluation']
    assert h.digest(ROOT/learning['source_run_manifest']) == learning['source_run_manifest_sha256']
    assert h.digest(ROOT/evaluation['api_manifest']) == evaluation['api_manifest_sha256']
    directory = ROOT/'research/runs'/args.run_id
    if directory.exists():
        raise ValueError('Preserve existing run; choose a new run ID.')
    artifacts = directory/'artifacts'; artifacts.mkdir(parents=True)
    models, ids, hashes = {}, None, {}
    for entry in learning['model_artifacts']:
        path = Path(args.rate_model_dir)/entry['id']
        assert h.digest(path) == entry['sha256']
        kind = entry['id'].removesuffix('.npz')
        with np.load(path,allow_pickle=False) as source:
            if ids is not None and not np.array_equal(ids,source['station_ids']):
                raise ValueError('Frozen model station order differs.')
            ids, models[kind] = source['station_ids'].copy(),source['means'].copy()
        hashes[kind] = entry['sha256']
    numeric = h.numeric_contract(np.concatenate([x.sum(-1).ravel() for x in models.values()]))
    cutoff = numeric['cutoff']
    contract_finished = datetime.now(timezone.utc).isoformat()
    print(json.dumps(dict(stage='historical_numeric_contract_before_api',**numeric)),flush=True)
    assert h.digest(args.panel) == evaluation['audited_panel']['sha256']
    with np.load(args.panel,allow_pickle=False) as source:
        data = h.pair_data(source['stock'],source['scheduled_at_unix_seconds'],source['observed_at_unix_seconds'],source['station_ids'],ids)
    if ((data['day_index'] < 0) | (data['day_index'] >= 14)).any():
        raise ValueError('Source calendar differs from frozen bootstrap dates.')
    shape = data['current'].shape
    predictions = {key:np.full(shape,np.nan) for key in ('mean','median','lower','upper','pzero','expected_mae_advantage')}
    crps = np.full(shape,np.nan)
    row, col = np.nonzero(np.isfinite(data['current']))
    rates = models['calendar'][data['weekday'][row],data['hour'][row],data['model_cols'][col]]
    for start in range(0,len(row),512):
        rr, cc = row[start:start+512],col[start:start+512]
        chosen = rates[start:start+512]
        # Predictor has only current stock and frozen historical rates; no labels.
        grid, pmf = state_pmf(data['current'][rr,cc],chosen[:,1],chosen[:,0],cutoff)
        outputs = summaries(grid,pmf)
        for key,value in outputs.items():
            predictions[key][rr,cc] = value
        # Labels are used only by the separate score function after prediction.
        crps[rr,cc] = integer_crps(grid,pmf,data['future'][rr,cc])
    groups = group_scores(data,predictions,crps)
    valid, truth = data['valid'],data['future']
    if not valid.any():
        raise ValueError('No valid official September evaluation pairs.')
    losses = {key:np.abs(value-truth) for key,value in [('mean',predictions['mean']),('median',predictions['median']),('persistence',data['current'].astype(float))]}
    daily = np.zeros((14,4))
    for day in range(14):
        mask = valid & (data['day_index'][:,None] == day)
        daily[day] = [losses['mean'][mask].sum(),losses['median'][mask].sum(),int(mask.sum()),losses['persistence'][mask].sum()]
    primary_blocks = [h.bootstrap(daily[:,:3],block) for block in (3,7)]
    secondary_blocks = [h.bootstrap(daily[:,[3,1,2]],block) for block in (3,7)]
    point = float((losses['mean'][valid]-losses['median'][valid]).mean())
    verdict = classify(data['scope']['matched_ids'],int(valid.sum()),int((daily[:,2] > 0).sum()),point,primary_blocks)
    metrics = dict(mean_to_median_mae_reduction=point,
        mean_mae=groups['all']['point_outputs']['mean']['mae'],median_mae=groups['all']['point_outputs']['median']['mae'],
        persistence_mae=groups['all']['point_outputs']['persistence']['mae'],
        persistence_to_median_mae_reduction=float((losses['persistence'][valid]-losses['median'][valid]).mean()),
        matched_training_ids=data['scope']['matched_ids'],valid_pairs=int(valid.sum()),valid_dates=int((daily[:,2] > 0).sum()))
    for item in primary_blocks:
        if item['interval'] is not None:
            metrics[f"ci_{item['block_days']}day_lower"],metrics[f"ci_{item['block_days']}day_upper"] = item['interval']
    for entry in learning['model_artifacts']:
        assert h.digest(Path(args.rate_model_dir)/entry['id']) == entry['sha256']
    np.savez_compressed(artifacts/'predictions.npz',**predictions,crps=crps,initial=data['current'],target=truth,
                        valid=valid,times=data['times'],model_cols=data['model_cols'])
    limitations = ['Same official September API reused after legacy work and H5; not untouched independent evaluation.',
        'Point-action diagnosis within one frozen distribution, not learned architecture/distribution/hyperparameter selection or evidence of improved probabilities.',
        'Effective historical completed-flow rates omit latent demand, exposure, capacity, operator movement and station changes.',
        'MAE-optimal action under the model need not be optimal for the true data distribution; mean/median cannot identify individual process errors.',
        'Conditional KST request clocks, daytime exact-ID paired observations Sep17..30, no October or missing-zero imputation.',
        'Block intervals condition on frozen historical weights and omit coefficient uncertainty and repeated-evaluation selection uncertainty.',
        'No operational adoption, probability recalibration, verified usability, policy effect or final GCN/FM architecture conclusion.']
    summary = dict(hypothesis_id='H-0006',decision=verdict,metrics=metrics,groups=groups,scope=data['scope'],
        numerical_contract=numeric,primary_bootstraps=primary_blocks,secondary_persistence_bootstraps=secondary_blocks,
        daily_scores=[dict(day_index=i,mean_loss=float(x[0]),median_loss=float(x[1]),pairs=int(x[2]),persistence_loss=float(x[3])) for i,x in enumerate(daily)],
        same_historical_distribution=True,api_training=False,api_selection_or_calibration=False,frozen_rate_hashes_unchanged=True,
        numerical_contract_before_api=contract_finished,elapsed_seconds=time.perf_counter()-timer,limitations=limitations)
    h.write(directory/'summary.json',summary)
    manifest = dict(schema_version=1,run_id=args.run_id,hypothesis_id='H-0006',plan_sha256=record['preregistration']['plan_sha256'],
        code_commit=code,input_manifest_sha256=h.digest(input_path),started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python','research_code/loss_consistent_stock.py','--panel','<PRIVATE_AUDITED_API_PANEL>','--rate-model-dir','<PRIVATE_FROZEN_HISTORY_MODELS>','--run-id',args.run_id],
        environment=dict(python=platform.python_version(),packages=dict(numpy=np.__version__,scipy=scipy.__version__),compute_location='dislab',device='CPU; no fitting',
            threads={x:os.environ.get(x) for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
        seed=20261031,evaluation_scope=data['scope'],metrics=metrics,frozen_model_hashes=hashes,
        artifacts=[dict(id='summary.json',sha256=h.digest(directory/'summary.json'),availability='public aggregate'),
                   dict(id='predictions.npz',sha256=h.digest(artifacts/'predictions.npz'),availability='private dislab prediction artifact; not published')],limitations=limitations)
    h.write(directory/'manifest.json',manifest)
    print(json.dumps(dict(stage='completed',decision=verdict,metrics=metrics,groups=groups,seconds=summary['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',required=True)
    ap.add_argument('--rate-model-dir',required=True)
    ap.add_argument('--run-id',required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception as error:
        directory = ROOT/'research/runs'/args.run_id
        if directory.exists() and not (directory/'summary.json').exists():
            h.write(directory/'attempt.json',dict(status='failed',decision='inconclusive',reason=type(error).__name__+': '+str(error)))
        raise
