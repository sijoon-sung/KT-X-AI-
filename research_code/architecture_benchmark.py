"""H-0002: time-blocked API-stock risk benchmark, never using transaction labels."""
import argparse
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def epoch(stamp):
    return int(datetime.fromisoformat(stamp).timestamp())


def origins(times, start, end):
    # No history or labels cross their own split boundary.
    return np.flatnonzero((times >= start + 3600) & (times < end - 1800)
                          & ((times + 32400) % 3600 == 0))


def station_mask(stock, times, start, end, ids, metadata):
    coverage = np.isfinite(stock[(times >= start) & (times < end)]).mean(0)
    known = np.array([s in metadata and np.isfinite(metadata[s]).all()
                      and abs(metadata[s][0]) <= 90 and abs(metadata[s][1]) <= 180 for s in ids])
    return known & (coverage >= .9)


def graph(coords, kind):
    n = len(coords)
    if kind == 'identity':
        return np.eye(n, dtype=np.float32)
    radians = np.deg2rad(coords)
    delta = radians[:, None, :] - radians[None, :, :]
    a = np.sin(delta[..., 0] / 2) ** 2 + (np.cos(radians[:, 0])[:, None]
        * np.cos(radians[:, 0])[None, :] * np.sin(delta[..., 1] / 2) ** 2)
    distance = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :8]
    result = np.eye(n, dtype=np.float32) * .5
    result[np.arange(n)[:, None], nearest] = .5 / 8
    if kind == 'shuffled':
        permutation = np.random.default_rng(20261031).permutation(n)
        result = result[np.ix_(permutation, permutation)]
    return result


def sparse_adjacency(array, device):
    row, col = np.nonzero(array)
    return torch.sparse_coo_tensor(np.stack([row, col]), array[row, col],
                                   array.shape, device=device).coalesce()


def mix(adj, h):
    b, n, d = h.shape
    return torch.sparse.mm(adj, h.permute(1, 0, 2).reshape(n, b * d)).reshape(n, b, d).permute(1, 0, 2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 32)
        self.ff = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, h, adj):
        h = h + F.gelu(mix(adj, self.linear(self.norm(h))))
        return h + self.ff(h)


class Context(nn.Module):
    def __init__(self, stations):
        super().__init__()
        self.embedding = nn.Embedding(stations, 8)
        self.project = nn.Linear(26, 32)
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x, adj):
        embedding = self.embedding.weight.unsqueeze(0).expand(len(x), -1, -1)
        h = F.gelu(self.project(torch.cat([x, embedding], -1)))
        for block in self.blocks:
            h = block(h, adj)
        return h


class Direct(nn.Module):
    def __init__(self, stations):
        super().__init__()
        self.context = Context(stations)
        self.head = nn.Linear(32, 1)

    def forward(self, x, adj):
        return self.head(self.context(x, adj)).squeeze(-1)


class Flow(nn.Module):
    def __init__(self, stations):
        super().__init__()
        self.context = Context(stations)
        self.project = nn.Linear(36, 32)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.head = nn.Linear(32, 1)

    def velocity(self, state, t, context, adj):
        features = torch.stack([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], -1)
        temporal = features[:, None, :].expand(-1, state.shape[1], -1)
        h = F.gelu(self.project(torch.cat([context, state[..., None], temporal], -1)))
        for block in self.blocks:
            h = block(h, adj)
        return self.head(h).squeeze(-1)

    def loss(self, x, y, valid, adj):
        context = self.context(x, adj)
        x0 = torch.randn_like(y)
        x1 = torch.log1p(y + torch.rand_like(y))
        # Unknown targets follow an unforced noise path and have zero loss weight.
        x1 = torch.where(valid, x1, x0)
        t = torch.rand(len(y), device=y.device)
        state = (1 - t[:, None]) * x0 + t[:, None] * x1
        error = (self.velocity(state, t, context, adj) - (x1 - x0)) ** 2
        return error[valid].mean()

    @torch.no_grad()
    def generate(self, x, adj, samples, chunk):
        context = self.context(x, adj)
        b, n, _ = context.shape
        generated = []
        for offset in range(0, samples, chunk):
            k = min(chunk, samples - offset)
            cond = context.repeat_interleave(k, dim=0)
            state = torch.randn(b * k, n, device=x.device)
            for step in range(12):
                t = torch.full((b * k,), step / 12, device=x.device)
                state = state + self.velocity(state, t, cond, adj) / 12
            generated.append(state.reshape(b, k, n))
        z = torch.cat(generated, 1)
        assert torch.isfinite(z).all(), 'Non-finite flow output.'
        p = (z < math.log(2)).float().mean(1)
        count = torch.floor(torch.expm1(z)).clamp_min(0)
        assert torch.isfinite(count).all(), 'Non-finite generated counts.'
        return p, count.mean(1), count.median(1).values

    @torch.no_grad()
    def zero_path(self, x, adj):
        context = self.context(x, adj)
        state = torch.zeros(x.shape[:2], device=x.device)
        for step in range(12):
            t = torch.full((len(x),), step / 12, device=x.device)
            state = state + self.velocity(state, t, context, adj) / 12
        return torch.floor(torch.expm1(state)).clamp_min(0)


def split_data(stock, times, actual, indices, geometry):
    history = stock[indices[:, None] - np.arange(6)[None, :]].transpose(0, 2, 1)
    mask = np.isfinite(history)
    local = times[indices] + 32400
    tod = (local % 86400) / 86400
    dow = ((local // 86400 + 3) % 7) / 7
    temporal = np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                         np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)], -1)
    b, n, _ = history.shape
    x = np.concatenate([np.log1p(np.where(mask, history, 0)) / 3, mask.astype(np.float32),
                        np.broadcast_to(temporal[:, None, :], (b, n, 4)),
                        np.broadcast_to(geometry[None, :, :], (b, n, 2))], -1).astype(np.float32)
    target = stock[indices + 3]
    elapsed = actual[indices + 3] - actual[indices]
    valid = mask.all(-1) & np.isfinite(target) & (elapsed >= 1680) & (elapsed <= 1920)
    return dict(x=torch.from_numpy(x), y=np.nan_to_num(target, nan=0), valid=valid,
                current=history[..., 0], day=(local // 86400).astype(int), origins=indices)


def daily_scores(p, data):
    truth = data['y'] == 0
    scored = (p - truth) ** 2
    result = {}
    for group, selection in [('all', np.ones_like(truth)), ('positive', data['current'] > 0),
                             ('empty', data['current'] == 0)]:
        values = []
        for day in np.unique(data['day']):
            chosen = data['valid'] & selection & (data['day'] == day)[:, None]
            values.append(float(scored[chosen].mean()) if chosen.any() else None)
        finite = [v for v in values if v is not None]
        result[group] = dict(mean=float(np.mean(finite)) if finite else None, daily=values)
    return result


def reliability(p, data):
    keep = data['valid']
    p, y = p[keep], (data['y'][keep] == 0)
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        bins.append(dict(bin=i, count=int(mask.sum()),
                         predicted=float(p[mask].mean()) if mask.any() else None,
                         observed=float(y[mask].mean()) if mask.any() else None))
    return bins


@torch.no_grad()
def predict(model, adj, data, device, seed, samples, chunk):
    model.eval()
    torch.manual_seed(seed)
    probabilities, means, medians, zeros = [], [], [], []
    for start in range(0, len(data['x']), 4):
        x = data['x'][start:start + 4].to(device)
        if isinstance(model, Flow):
            p, mean, median = model.generate(x, adj, samples, chunk)
            means.append(mean.cpu().numpy())
            medians.append(median.cpu().numpy())
            zeros.append(model.zero_path(x, adj).cpu().numpy())
        else:
            p = model(x, adj).sigmoid()
        probabilities.append(p.cpu().numpy())
    return (np.concatenate(probabilities),
            dict(mean=np.concatenate(means), median=np.concatenate(medians), zero=np.concatenate(zeros))
            if means else {})


def interval(difference):
    difference = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(20261031)
    n = len(difference)
    starts = rng.integers(0, n - 1, (2000, (n + 1) // 2))
    indices = (starts[..., None] + np.arange(2)).reshape(2000, -1)[:, :n]
    return np.percentile(np.nanmean(difference[indices], 1), [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--sample-chunk', type=int, default=16)
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    use_cuda = torch.cuda.is_available() if args.device == 'auto' else args.device == 'cuda'
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(.15, .55 * free / total))
    record = json.loads((ROOT / 'research/hypotheses/H-0002.json').read_text(encoding='utf-8'))
    canonical = json.dumps(record['plan'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == record['preregistration']['plan_sha256']
    assert record['status'] in ('preregistered', 'running')
    manifest_path = ROOT / 'research/datasets/tashu-primary-v1.json'
    assert digest(manifest_path) == record['plan']['data']['input_manifest_sha256']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    prepared = Path(args.prepared)
    panel_path = prepared / 'data/observed_panel.npz'
    assert digest(panel_path) == manifest['derived_files'][0]['sha256']
    meta_path = prepared / 'raw/meta/stations_20260917.json.gz'
    expected = next(f['sha256'] for f in manifest['files'] if f['id'].endswith('stations_20260917.json.gz'))
    assert digest(meta_path) == expected
    with gzip.open(meta_path, 'rt', encoding='utf-8') as stream:
        metadata = {}
        for station in json.load(stream)['results']:
            try:
                coordinates = [float(station['x_pos']), float(station['y_pos'])]
            except (ValueError, TypeError, KeyError):
                coordinates = [float('nan'), float('nan')]
            metadata[str(station['id'])] = coordinates
    with np.load(panel_path, allow_pickle=False) as panel:
        stock, times = panel['stock'], panel['scheduled_at_unix_seconds']
        actual, ids = panel['observed_at_unix_seconds'], panel['station_ids'].astype(str)
    periods = record['plan']['evaluation']['periods']
    keep = station_mask(stock, times, epoch(periods[0]['start']), epoch(periods[0]['end']), ids, metadata)
    selected_ids = ids[keep]
    stock, actual = stock[:, keep], actual[:, keep]
    coords = np.array([metadata[s] for s in selected_ids], dtype=np.float32)
    geometry = (coords - coords.mean(0)) / np.maximum(coords.std(0), 1e-6)
    splits = {p['role']: split_data(stock, times, actual,
              origins(times, epoch(p['start']), epoch(p['end'])), geometry) for p in periods}
    n = len(selected_ids)
    print(json.dumps(dict(device=str(device), stations=n,
          split_origins={k: len(v['x']) for k, v in splits.items()},
          split_valid_cells={k: int(v['valid'].sum()) for k, v in splits.items()})), flush=True)
    adjs = {kind: sparse_adjacency(graph(coords, kind), device) for kind in ('identity', 'geo', 'shuffled')}
    run_dir = ROOT / 'research/runs' / args.run_id
    private = run_dir / 'artifacts'
    private.mkdir(parents=True, exist_ok=True)
    test, train, validation = splits['test'], splits['train'], splits['validation']
    records = []
    for name in ('persistence', 'frequency'):
        p = (test['current'] == 0).astype(float)
        if name == 'frequency':
            rates = []
            for positive in (False, True):
                group = (train['current'] > 0) if positive else (train['current'] == 0)
                valid = train['valid'] & group
                counts = valid.sum(0)
                zeros = ((train['y'] == 0) & valid).sum(0)
                prior = zeros.sum() / max(1, counts.sum())
                rates.append((zeros + 20 * prior) / (counts + 20))
            p = np.where(test['current'] > 0, rates[1], rates[0])
        records.append(dict(model=name, seed=None, test=daily_scores(p, test),
                            reliability=reliability(p, test)))
    for kind, model_class in [('identity', Direct), ('geo', Direct), ('shuffled', Direct), ('cfm', Flow)]:
        adj = adjs['geo' if kind == 'cfm' else kind]
        for seed in SEEDS:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = model_class(n).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            best, selected, best_state = np.inf, None, None
            trace = []
            start = time.perf_counter()
            for update in range(1, 601):
                model.train()
                batch = rng.integers(0, len(train['x']), size=4)
                x = train['x'][batch].to(device)
                y = torch.as_tensor(train['y'][batch], dtype=torch.float32, device=device)
                valid = torch.as_tensor(train['valid'][batch], device=device)
                assert valid.any(), 'No valid targets in training batch.'
                optimizer.zero_grad(set_to_none=True)
                if kind == 'cfm':
                    loss = model.loss(x, y, valid, adj)
                else:
                    loss = F.binary_cross_entropy_with_logits(model(x, adj)[valid], (y == 0).float()[valid])
                assert torch.isfinite(loss), 'Non-finite training loss.'
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                if update % 100 == 0:
                    # Preserve the training stream across validation sampling.
                    state = torch.random.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state_all() if use_cuda else None
                    p, _ = predict(model, adj, validation, device, seed + 10000, 64, args.sample_chunk)
                    torch.random.set_rng_state(state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state_all(cuda_state)
                    score = daily_scores(p, validation)['positive']['mean']
                    assert score is not None and np.isfinite(score)
                    trace.append(dict(update=update, validation_brier=score))
                    if score < best:
                        best, selected = score, update
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(f'{kind} seed={seed} update={update} validation_brier={score:.6f}', flush=True)
            training_seconds = time.perf_counter() - start
            model.load_state_dict(best_state)
            torch.save(best_state, private / f'{kind}-{seed}.pt')
            start = time.perf_counter()
            p, counts = predict(model, adj, test, device, seed + 20000, 128, args.sample_chunk)
            inference_seconds = time.perf_counter() - start
            row = dict(model=kind, seed=seed, parameters=sum(t.numel() for t in model.parameters()),
                       selected_update=selected, validation_brier=best, trace=trace,
                       training_and_validation_seconds=training_seconds,
                       test_inference_seconds=inference_seconds, test=daily_scores(p, test),
                       reliability=reliability(p, test))
            if kind == 'cfm':
                valid = test['valid'] & (test['current'] > 0)
                correction = p * (1 - p) / 127
                row['mc_corrected_brier_current_positive'] = float(
                    np.mean([(((p - (test['y'] == 0)) ** 2 - correction)[valid & (test['day'] == d)[:, None]]).mean()
                             for d in np.unique(test['day'])]))
                row['count_mae'] = {key: float(np.abs(value[test['valid']] - test['y'][test['valid']]).mean())
                                    for key, value in counts.items()}
                row['persistence_count_mae'] = float(np.abs(test['current'][test['valid']] - test['y'][test['valid']]).mean())
                row['zero_start_brier'] = daily_scores((counts['zero'] == 0).astype(float), test)
            np.savez_compressed(private / f'{kind}-{seed}-test.npz', p=p)
            records.append(row)
            write(private / 'progress.json', records)
            print(json.dumps(dict(model=kind, seed=seed, test_brier=row['test']['positive']['mean'],
                                  seconds=training_seconds + inference_seconds)), flush=True)
            del model, optimizer, best_state
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
    comparisons = []
    flow_daily = np.mean([r['test']['positive']['daily'] for r in records if r['model'] == 'cfm'], 0)
    for name in ('identity', 'geo'):
        control = np.mean([r['test']['positive']['daily'] for r in records if r['model'] == name], 0)
        delta = control - flow_daily
        comparisons.append(dict(control=name, relative_improvement=float(delta.mean() / control.mean()),
                                absolute_improvement=float(delta.mean()), ci=interval(delta)))
    if np.isfinite(flow_daily).sum() < 5:
        verdict = 'inconclusive'
    elif all(c['relative_improvement'] >= .02 and c['ci'][0] > 0 for c in comparisons):
        verdict = 'supported'
    elif any(c['relative_improvement'] <= 0 or c['ci'][1] <= 0 for c in comparisons):
        verdict = 'not_supported'
    else:
        verdict = 'inconclusive'
    metrics = dict(brier_current_positive=float(flow_daily.mean()))
    for c in comparisons:
        metrics[c['control'] + '_relative_improvement'] = c['relative_improvement']
    limitations = ['Single conditional-KST, one-season, one-city retrospective benchmark with seven test dates.',
                   'Test dates were previously included in H-0001 observation summaries; not wholly unseen raw data.',
                   'Small stock-adapted models, fixed update budget; not a reproduction of the original full Transit-CFM.',
                   'CFM has a different training objective and more parameters; Monte Carlo plug-in Brier includes sampling noise.',
                   'Observed future API zero is not latent demand, verified bike usability or policy effectiveness.']
    summary = dict(hypothesis_id='H-0002', decision=verdict, device=str(device), stations=n,
                   station_selection_sha256=hashlib.sha256('\n'.join(selected_ids).encode()).hexdigest(),
                   scope={k: dict(origins=len(v['x']), valid_cells=int(v['valid'].sum()),
                                  current_positive_cells=int((v['valid'] & (v['current'] > 0)).sum()),
                                  dates=len(np.unique(v['day']))) for k, v in splits.items()},
                   metrics=metrics, comparisons=comparisons, records=records, limitations=limitations)
    write(run_dir / 'summary.json', summary)
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    run_manifest = dict(schema_version=1, run_id=args.run_id, hypothesis_id='H-0002',
        plan_sha256=record['preregistration']['plan_sha256'], code_commit=code_commit,
        input_manifest_sha256=digest(manifest_path), started_at_utc=started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python', 'research_code/architecture_benchmark.py', '--prepared', '<PRIVATE_AUDITED_INPUT>',
                 '--run-id', args.run_id, '--device', args.device, '--sample-chunk', str(args.sample_chunk)],
        environment=dict(python=platform.python_version(), packages=dict(numpy=np.__version__, torch=torch.__version__),
                         device=str(device), gpu=torch.cuda.get_device_name() if use_cuda else None,
                         cuda=torch.version.cuda, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if use_cuda else 0,
                         cpu_threads=4), seed=list(SEEDS), evaluation_scope=summary['scope'], metrics=metrics,
        artifacts=[dict(id='summary.json', sha256=digest(run_dir / 'summary.json'), availability='public aggregate'),
                   dict(id='checkpoints and individual probabilities', availability='private local artifacts')],
        limitations=limitations)
    write(run_dir / 'manifest.json', run_manifest)
    print(json.dumps(dict(decision=verdict, metrics=metrics, comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    main()
