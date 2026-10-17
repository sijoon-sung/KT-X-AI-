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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def graph_tensor(graph, device):
    tensor = torch.as_tensor(graph, dtype=torch.float32, device=device)
    return tensor.to_sparse_coo().coalesce() if len(graph) >= 64 else tensor


def graph_multiply(graph, values):
    if graph.is_sparse:
        batch, nodes, width = values.shape
        flat = values.permute(1, 0, 2).reshape(nodes, batch * width)
        # Sparse AMP support differs by torch/CUDA build. Accumulate this operation
        # in float32 and return to the surrounding activation dtype.
        with torch.autocast(values.device.type, enabled=False):
            result = torch.sparse.mm(graph.float(), flat.float())
        return result.to(values.dtype).reshape(nodes, batch, width).permute(1, 0, 2)
    return torch.matmul(graph, values)


class GraphEncoder(nn.Module):
    def __init__(self, features, hidden, layers=2, dropout=0.):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph):
        h = nn.functional.silu(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + self.dropout(nn.functional.silu(layer(graph_multiply(graph, h)))))
        return h


class CountGCN(nn.Module):
    def __init__(self, features, hidden, channels, layers=2, dropout=0.):
        super().__init__()
        self.encoder = GraphEncoder(features, hidden, layers, dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, channels))

    def forward(self, x, graph):
        return self.from_context(self.encoder(x, graph))

    def from_context(self, context):
        return nn.functional.softplus(self.head(context)) + 1e-5


class ResidualFlowBlock(nn.Module):
    """Residual spatial message + expanded FFN + per-block time conditioning."""
    def __init__(self, hidden, expansion, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.message = nn.Linear(hidden, hidden)
        self.time_modulation = nn.Linear(3, hidden * 2)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * expansion), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden * expansion, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, clock, graph):
        scale, shift = self.time_modulation(clock).chunk(2, -1)
        state = self.norm1(h) * (1 + .1 * torch.tanh(scale)) + .1 * shift
        h = h + self.dropout(nn.functional.silu(self.message(graph_multiply(graph, state))))
        return h + self.dropout(self.ffn(self.norm2(h)))


class GraphFlow(nn.Module):
    def __init__(self, base, hidden, channels, blocks=1, expansion=2, dropout=0.):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.base.eval()
        self.input = nn.Linear(channels + hidden + 3, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        # Empty for legacy/base/medium; existing checkpoint key names remain valid.
        self.blocks = nn.ModuleList([ResidualFlowBlock(hidden, expansion, dropout) for _ in range(blocks - 1)])

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Frozen encoder dropout must also stay frozen.
        return self

    def velocity(self, residual, time, context, graph):
        clock = torch.stack([time, torch.sin(time * torch.pi), torch.cos(time * torch.pi)], -1)
        clock = clock[:, None].expand(-1, residual.shape[1], -1)
        h = nn.functional.silu(self.input(torch.cat([residual, context, clock], -1)))
        h = self.norm(h + nn.functional.silu(self.message(graph_multiply(graph, h))))
        for block in self.blocks:
            h = block(h, clock, graph)
        return self.output(h)

    def objective(self, x, y, mask, graph, generator=None):
        context = self.base.encoder(x, graph).detach()
        baseline = torch.log1p(self.base.from_context(context).float()).detach()
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
    def sample(self, x, graph, samples, steps, generator, chunk_size=None, amp=False):
        self.eval()
        b, n, _ = x.shape
        if samples < 1 or steps < 1 or (chunk_size is not None and chunk_size < 1):
            raise ValueError("Samples, steps and chunk size must be positive.")
        use_amp = bool(amp and x.device.type == "cuda")
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_amp):
            original_context = self.base.encoder(x, graph)
            original_baseline = torch.log1p(self.base.from_context(original_context).float())
        channels = original_baseline.shape[-1]
        # The random prior is small; drawing it once keeps the same starts when
        # chunk size changes. Wide hidden activations are only allocated per chunk.
        starts = torch.randn(samples, b, n, channels, device=x.device, generator=generator) * .5
        output = torch.empty(samples, b, n, channels, dtype=torch.float32, device="cpu")
        chunk_size = samples if chunk_size is None else min(chunk_size, samples)
        dt = 1.0 / steps
        for offset in range(0, samples, chunk_size):
            count = min(chunk_size, samples - offset)
            context = original_context.repeat(count, 1, 1)
            baseline = original_baseline.repeat(count, 1, 1)
            state = starts[offset:offset + count].reshape(count * b, n, channels).clone()
            with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_amp):
                for step in range(steps):
                    t = torch.full((len(state),), step * dt, device=x.device)
                    v1 = self.velocity(state, t, context, graph)
                    v2 = self.velocity(state + .5 * dt * v1, t + .5 * dt, context, graph)
                    state = state + dt * v2.float()
            # Numerical guard in log space; not a station-capacity constraint.
            log_count = (state.float() + baseline).clamp(-12, 8)
            output[offset:offset + count] = torch.floor(torch.expm1(log_count).clamp_min(0)).reshape(count, b, n, channels).cpu()
        return output


def poisson_objective(pred, y, mask):
    pred = pred.float()
    return (pred - y * torch.log(pred))[mask].mean()


def train_neural(model, x, y, mask, train, validation, graph, epochs, batch_size, lr, seed, flow=False,
                 accumulate=1, amp=False, patience=0):
    device = next(model.parameters()).device
    # Aggressively exploit GPU VRAM: if using CUDA and data fits (< 1.5GB),
    # preload tensors directly onto GPU to eliminate repeated PCIe host-to-device transfers.
    data_bytes = getattr(x, "nbytes", 0) + getattr(y, "nbytes", 0) + getattr(mask, "nbytes", 0)
    if device.type == "cuda" and data_bytes < 1024 * 1024 * 1536:
        x_dev = torch.as_tensor(x, dtype=torch.float32, device=device)
        y_dev = torch.as_tensor(y, dtype=torch.float32, device=device)
        m_dev = torch.as_tensor(mask, dtype=torch.bool, device=device)
        get_batch = lambda idx: (x_dev[idx], y_dev[idx], m_dev[idx])
    else:
        get_batch = lambda idx: tuple(torch.as_tensor(arr[idx], device=device) for arr in (x, y, mask))
    graph = graph_tensor(graph, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    if accumulate < 1 or patience < 0:
        raise ValueError("Accumulate must be positive; patience must be nonnegative.")
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    best, best_state, logs, stale = float("inf"), None, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss, training_cells = 0., 0
        order = rng.permutation(train)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for offset in range(0, len(train), batch_size * accumulate):
            group = order[offset:offset + batch_size * accumulate]
            group_cells = int(mask[group].sum())
            if not group_cells:
                continue
            optimizer.zero_grad()
            for micro in range(0, len(group), batch_size):
                idx = group[micro:micro + batch_size]
                a, b, m = get_batch(idx)
                count = int(mask[idx].sum())
                if not count:
                    continue
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    loss = model.objective(a, b, m, graph) if flow else poisson_objective(model(a, graph), b, m)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; inspect input ranges or use --no-amp.")
                scaler.scale(loss * (count / group_cells)).backward()
                weighted_loss += float(loss.detach()) * count
                training_cells += count
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        total, cells = 0.0, 0
        # Fixed validation noise makes checkpoint selection comparable across epochs.
        gen = torch.Generator(device=device).manual_seed(seed + 1000)
        with torch.no_grad():
            for offset in range(0, len(validation), batch_size):
                idx = validation[offset:offset + batch_size]
                a, b, m = get_batch(idx)
                if not m.any():
                    continue
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    value = model.objective(a, b, m, graph, gen) if flow else poisson_objective(model(a, graph), b, m)
                count = int(m.sum())
                total += float(value) * count
                cells += count
        if not training_cells or not cells:
            raise ValueError("Training and validation require observed target cells.")
        score = total / cells
        logs.append({"epoch": epoch, "train_loss": weighted_loss / training_cells, "validation_loss": score,
                     "amp": use_amp, "effective_batch_size": batch_size * accumulate,
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"{'flow' if flow else 'gcn'} epoch {epoch}/{epochs}: validation={score:.4f}", flush=True)
        if patience and stale >= patience:
            print(f"Early stop: {patience} epochs without validation improvement", flush=True)
            break
    if best_state is None:
        raise ValueError("No finite validation checkpoint.")
    model.load_state_dict(best_state)  # Saved and evaluated model are identical.
    return logs


def tabular_features(x):
    b, n, f = x.shape
    station = np.broadcast_to(np.arange(n)[None, :, None] / max(n - 1, 1), (b, n, 1))
    return np.concatenate([x, station], -1).reshape(b * n, f + 1)


def train_xgboost(x, y, mask, train, validation, seed, trees, depth=4, device="cpu"):
    train_x, val_x = tabular_features(x[train]), tabular_features(x[validation])
    train_y, val_y = y[train].reshape(-1, y.shape[-1]), y[validation].reshape(-1, y.shape[-1])
    train_m, val_m = mask[train].reshape(train_y.shape), mask[validation].reshape(val_y.shape)
    models = []
    for channel in range(y.shape[-1]):
        if not train_m[:, channel].any() or not val_m[:, channel].any():
            raise ValueError("Each horizon/channel needs observed training and validation targets.")
        model = XGBRegressor(objective="count:poisson", n_estimators=trees, max_depth=depth, device=device,
                             learning_rate=.05, subsample=.9, colsample_bytree=.9,
                             tree_method="hist", n_jobs=4, random_state=seed,
                             base_score=max(float(train_y[train_m[:, channel], channel].mean()), .01),
                             early_stopping_rounds=15)
        model.fit(train_x[train_m[:, channel]], train_y[train_m[:, channel], channel],
                  eval_set=[(val_x[val_m[:, channel]], val_y[val_m[:, channel], channel])], verbose=False)
        actual_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if device == "cuda" and not actual_device.startswith("cuda"):
            raise ValueError("XGBoost silently fell back to CPU. Install a CUDA-enabled build or set --xgb-device cpu explicitly.")
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
    gcn = CountGCN(*args, layers=metadata.get("gcn_layers", 2), dropout=metadata.get("dropout", 0.)).to(device)
    gcn.load_state_dict(torch.load(directory / "gcn.pt", map_location=device, weights_only=True))
    flow = GraphFlow(gcn, metadata["hidden"], metadata["horizon"] * 2,
                     blocks=metadata.get("flow_blocks", 1), expansion=metadata.get("expansion", 2),
                     dropout=metadata.get("dropout", 0.)).to(device)
    flow.load_state_dict(torch.load(directory / "flow.pt", map_location=device, weights_only=True))
    xgb = []
    for channel in range(metadata["horizon"] * 2):
        model = XGBRegressor()
        model.load_model(directory / f"xgb_{channel}.json")
        # Tabular inference uses NumPy/CPU; avoid implicit cross-device fallback.
        model.set_params(device="cpu")
        xgb.append(model)
    gcn.eval()
    flow.eval()
    return xgb, gcn, flow


@torch.no_grad()
def predict_scenarios(name, models, x, graph, horizon, samples, steps, seed, sample_chunk=None, amp=False):
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
        with torch.autocast(device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            mean = gcn(torch.as_tensor(x, device=device), graph_tensor(graph, device)).float().cpu().numpy()
        values = rng.poisson(mean.clip(0, 1e5), size=(samples, *mean.shape))
    elif name == "gcn_flow":
        gen = torch.Generator(device=device).manual_seed(seed)
        values = flow.sample(torch.as_tensor(x, device=device), graph_tensor(graph, device),
                             samples, steps, gen, chunk_size=sample_chunk, amp=amp).numpy()
    else:
        raise ValueError(f"Unknown model: {name}")
    return values.reshape(samples, len(x), x.shape[1], horizon, 2).transpose(0, 1, 3, 2, 4).astype(np.float32)
