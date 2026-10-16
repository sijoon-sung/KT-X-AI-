"""Comparable count forecasts and a conditional graph residual flow.

GCN is fitted with a Poisson objective. Flow uses the same trained encoder and
mean, freezes them, then fits joint log-count residual scenarios. Uniform
dequantization followed by floor(expm1(.)) retains an explicit zero-count region.
This is independent-coupling conditional FM, not minibatch optimal transport.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from xgboost import XGBRegressor


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def graph_tensor(graph, device):
    tensor = torch.as_tensor(graph, dtype=torch.float32, device=device)
    return tensor.to_sparse_coo().coalesce() if len(graph) >= 64 else tensor


def graph_multiply(graph, values):
    if graph.is_sparse:
        batch, nodes, width = values.shape
        flat = values.permute(1, 0, 2).reshape(nodes, batch * width)
        return torch.sparse.mm(graph, flat).reshape(nodes, batch, width).permute(1, 0, 2)
    return torch.matmul(graph, values)


class GraphEncoder(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])

    def forward(self, x, graph):
        h = nn.functional.silu(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + nn.functional.silu(layer(graph_multiply(graph, h))))
        return h


class CountGCN(nn.Module):
    def __init__(self, features, hidden, channels):
        super().__init__()
        self.encoder = GraphEncoder(features, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, channels))

    def forward(self, x, graph):
        return nn.functional.softplus(self.head(self.encoder(x, graph))) + 1e-5


class GraphFlow(nn.Module):
    def __init__(self, base, hidden, channels):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.input = nn.Linear(channels + hidden + 3, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, channels))

    def velocity(self, residual, time, context, graph):
        clock = torch.stack([time, torch.sin(time * torch.pi), torch.cos(time * torch.pi)], -1)
        clock = clock[:, None].expand(-1, residual.shape[1], -1)
        h = nn.functional.silu(self.input(torch.cat([residual, context, clock], -1)))
        h = self.norm(h + nn.functional.silu(self.message(graph_multiply(graph, h))))
        return self.output(h)

    def objective(self, x, y, mask, graph, generator=None):
        context = self.base.encoder(x, graph).detach()
        baseline = torch.log1p(self.base(x, graph)).detach()
        noise = torch.rand(y.shape, device=y.device, generator=generator)
        target = torch.log1p(y + noise) - baseline
        # Missing target coordinates are not secretly presented to other nodes.
        # They stay on the independent noise prior and have zero target velocity.
        start = torch.randn(y.shape, device=y.device, generator=generator) * .5
        target = torch.where(mask, target, start)
        time = torch.rand(len(x), device=x.device, generator=generator)
        state = (1 - time[:, None, None]) * start + time[:, None, None] * target
        error = (self.velocity(state, time, context, graph) - (target - start)) ** 2
        return error[mask].mean()

    @torch.no_grad()
    def sample(self, x, graph, samples, steps, generator):
        self.eval()
        b, n, _ = x.shape
        context = self.base.encoder(x, graph).repeat(samples, 1, 1)
        baseline = torch.log1p(self.base(x, graph)).repeat(samples, 1, 1)
        state = torch.randn(b * samples, n, baseline.shape[-1], device=x.device, generator=generator) * .5
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((len(state),), step * dt, device=x.device)
            v1 = self.velocity(state, t, context, graph)
            v2 = self.velocity(state + .5 * dt * v1, t + .5 * dt, context, graph)
            state = state + dt * v2
        # Numerical guard in log space; not a station-capacity constraint.
        log_count = (state + baseline).clamp(-12, 8)
        return torch.floor(torch.expm1(log_count).clamp_min(0)).reshape(samples, b, n, -1)


def poisson_objective(pred, y, mask):
    return (pred - y * torch.log(pred))[mask].mean()


def train_neural(model, x, y, mask, train, validation, graph, epochs, batch_size, lr, seed, flow=False):
    device = next(model.parameters()).device
    # Keep the full data on CPU, moving batches only; supports larger real panels.
    graph = graph_tensor(graph, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_state, logs = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for offset in range(0, len(train), batch_size):
            # One permutation per epoch, not one new permutation per batch.
            if offset == 0:
                order = rng.permutation(train)
            idx = order[offset:offset + batch_size]
            a = torch.as_tensor(x[idx], device=device)
            b = torch.as_tensor(y[idx], device=device)
            m = torch.as_tensor(mask[idx], device=device)
            if not m.any():
                continue
            loss = model.objective(a, b, m, graph) if flow else poisson_objective(model(a, graph), b, m)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; inspect input ranges.")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        total, cells = 0.0, 0
        # Fixed validation noise makes checkpoint selection comparable across epochs.
        gen = torch.Generator(device=device).manual_seed(seed + 1000)
        with torch.no_grad():
            for offset in range(0, len(validation), batch_size):
                idx = validation[offset:offset + batch_size]
                a, b, m = (torch.as_tensor(arr[idx], device=device) for arr in (x, y, mask))
                if not m.any():
                    continue
                value = model.objective(a, b, m, graph, gen) if flow else poisson_objective(model(a, graph), b, m)
                count = int(m.sum())
                total += float(value) * count
                cells += count
        score = total / max(cells, 1)
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": score})
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"{'flow' if flow else 'gcn'} epoch {epoch}/{epochs}: validation={score:.4f}", flush=True)
    if best_state is None:
        raise ValueError("No finite validation checkpoint.")
    model.load_state_dict(best_state)  # Saved and evaluated model are identical.
    return logs


def tabular_features(x):
    b, n, f = x.shape
    station = np.broadcast_to(np.arange(n)[None, :, None] / max(n - 1, 1), (b, n, 1))
    return np.concatenate([x, station], -1).reshape(b * n, f + 1)


def train_xgboost(x, y, mask, train, validation, seed, trees):
    train_x, val_x = tabular_features(x[train]), tabular_features(x[validation])
    train_y, val_y = y[train].reshape(-1, y.shape[-1]), y[validation].reshape(-1, y.shape[-1])
    train_m, val_m = mask[train].reshape(train_y.shape), mask[validation].reshape(val_y.shape)
    models = []
    for channel in range(y.shape[-1]):
        if not train_m[:, channel].any() or not val_m[:, channel].any():
            raise ValueError("Each horizon/channel needs observed training and validation targets.")
        model = XGBRegressor(objective="count:poisson", n_estimators=trees, max_depth=4,
                             learning_rate=.05, subsample=.9, colsample_bytree=.9,
                             tree_method="hist", n_jobs=4, random_state=seed,
                             base_score=max(float(train_y[train_m[:, channel], channel].mean()), .01),
                             early_stopping_rounds=15)
        model.fit(train_x[train_m[:, channel]], train_y[train_m[:, channel], channel],
                  eval_set=[(val_x[val_m[:, channel]], val_y[val_m[:, channel], channel])], verbose=False)
        models.append(model)
    return models


def save_models(directory, xgb, gcn, flow, logs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for channel, model in enumerate(xgb):
        model.save_model(directory / f"xgb_{channel}.json")
    torch.save(gcn.state_dict(), directory / "gcn.pt")
    torch.save(flow.state_dict(), directory / "flow.pt")
    (directory / "training.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")


def load_models(directory, metadata, device):
    directory = Path(directory)
    args = (metadata["feature_count"], metadata["hidden"], metadata["horizon"] * 2)
    gcn = CountGCN(*args).to(device)
    gcn.load_state_dict(torch.load(directory / "gcn.pt", map_location=device, weights_only=True))
    flow = GraphFlow(gcn, metadata["hidden"], metadata["horizon"] * 2).to(device)
    flow.load_state_dict(torch.load(directory / "flow.pt", map_location=device, weights_only=True))
    xgb = []
    for channel in range(metadata["horizon"] * 2):
        model = XGBRegressor()
        model.load_model(directory / f"xgb_{channel}.json")
        xgb.append(model)
    gcn.eval()
    flow.eval()
    return xgb, gcn, flow


@torch.no_grad()
def predict_scenarios(name, models, x, graph, horizon, samples, steps, seed):
    """Return [S,B,H,N,2]. GCN/XGB use explicit independent Poisson baselines."""
    xgb, gcn, flow = models
    device = next(gcn.parameters()).device
    rng = np.random.default_rng(seed)
    if name == "xgboost":
        flat = tabular_features(x)
        mean = np.stack([m.predict(flat) for m in xgb], -1).reshape(len(x), x.shape[1], -1).clip(0, 1e5)
        values = rng.poisson(mean, size=(samples, *mean.shape))
    elif name == "gcn":
        gcn.eval()
        mean = gcn(torch.as_tensor(x, device=device), graph_tensor(graph, device)).cpu().numpy()
        values = rng.poisson(mean.clip(0, 1e5), size=(samples, *mean.shape))
    elif name == "gcn_flow":
        gen = torch.Generator(device=device).manual_seed(seed)
        values = flow.sample(torch.as_tensor(x, device=device), graph_tensor(graph, device),
                             samples, steps, gen).cpu().numpy()
    else:
        raise ValueError(f"Unknown model: {name}")
    return values.reshape(samples, len(x), x.shape[1], horizon, 2).transpose(0, 1, 3, 2, 4).astype(np.float32)
