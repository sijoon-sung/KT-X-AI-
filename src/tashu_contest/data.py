"""Explicit hourly observations; missing counts never become training labels of zero.

At time t we predict [t,t+H). Counts in that window are targets; only counts
strictly before t and a measured opening stock at t can enter the features.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def local_times(values):
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    if not (parsed == parsed.dt.floor("h")).all():
        raise ValueError("timestamp must be an hour boundary (KST).")
    return parsed


@dataclass
class Panel:
    times: pd.DatetimeIndex
    stations: list[str]
    counts: np.ndarray  # [T,N,2], NaN means not observed
    stock: np.ndarray   # stock at START of each hour, not end
    coords: np.ndarray  # [N,2] latitude, longitude
    capacity: np.ndarray
    names: list[str]
    requested_rentals: np.ndarray | None = None  # Optional replay-only evidence; never a predictor input.


def read_panel(path: str | Path) -> Panel:
    frame = pd.read_csv(path, dtype={"station_id": str}, encoding="utf-8-sig")
    required = {"timestamp", "station_id", "rentals", "returns", "lat", "lon"}
    if missing := required - set(frame):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame.station_id.isna().any():
        raise ValueError("Panel must contain valid station IDs and rows.")
    frame["timestamp"] = local_times(frame.timestamp)
    if frame.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("Duplicate timestamp/station_id rows.")
    for col in ("rentals", "returns", "stock", "capacity", "lat", "lon", "requested_rentals"):
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        vals = frame[col].dropna().to_numpy()
        if not np.isfinite(vals).all():
            raise ValueError(f"{col} must be finite or blank.")
        if col not in ("lat", "lon") and ((vals < 0).any() or (vals != np.floor(vals)).any()):
            raise ValueError(f"{col} must be a nonnegative integer or blank.")
    if frame.timestamp.isna().any():
        raise ValueError("Missing timestamps.")
    if (frame.requested_rentals < frame.rentals).any():
        raise ValueError("requested_rentals cannot be smaller than served rentals in the same hour.")
    stations = sorted(frame.station_id.unique())
    meta = []
    for sid in stations:
        rows = frame[frame.station_id == sid]
        for col in ("lat", "lon", "capacity"):
            if rows[col].nunique(dropna=True) > 1:
                raise ValueError(f"{sid}: {col} changes; split versions explicitly.")
        first = rows.iloc[0]
        lat, lon = first.lat, first.lon
        if not (np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{sid}: valid coordinates required.")
        cap = rows.capacity.dropna()
        cap = float(cap.iloc[0]) if len(cap) else np.nan
        if np.isfinite(cap) and (rows.stock.dropna() > cap).any():
            raise ValueError(f"{sid}: stock exceeds capacity; correct capacity or leave it blank.")
        name = str(first.get("name", sid))
        meta.append((lat, lon, cap, sid if name == "nan" else name))
    times = pd.date_range(frame.timestamp.min(), frame.timestamp.max(), freq="h")
    if len(times) * len(stations) > 100_000_000:
        raise ValueError("Panel is too large; check timestamps or split the input period.")
    index = pd.MultiIndex.from_product([times, stations], names=["timestamp", "station_id"])
    dense = frame.set_index(["timestamp", "station_id"]).reindex(index)
    shape = (len(times), len(stations))
    counts = np.stack([dense[c].to_numpy().reshape(shape) for c in ("rentals", "returns")], -1)
    return Panel(times, stations, counts.astype(np.float32),
                 dense.stock.to_numpy().reshape(shape).astype(np.float32),
                 np.array([r[:2] for r in meta], dtype=np.float64),
                 np.array([r[2] for r in meta]), [r[3] for r in meta],
                 dense.requested_rentals.to_numpy().reshape(shape).astype(np.float32)
                 if dense.requested_rentals.notna().any() else None)


def make_features(panel: Panel, history: int, origins=None) -> np.ndarray:
    """Causal features [origins,N,F]; build only requested windows to bound RAM.

    Sparse lags 1/2/3/24/168 (up to history), plus the history endpoint.
    """
    origins = np.arange(len(panel.times)) if origins is None else np.asarray(origins)
    t_count, n_count = len(origins), len(panel.stations)
    pieces = []
    lags = sorted({lag for lag in (1, 2, 3, 24, 168, history) if 0 < lag <= history})
    for lag in lags:
        prior = panel.counts[np.maximum(origins - lag, 0)].copy()
        prior[origins < lag] = np.nan
        pieces.extend([np.log1p(np.nan_to_num(prior, nan=0.0)), np.isfinite(prior).astype(np.float32)])
    pieces += [np.log1p(np.nan_to_num(panel.stock[origins], nan=0.0))[..., None],
               np.isfinite(panel.stock[origins]).astype(np.float32)[..., None]]
    hour = panel.times[origins].hour.to_numpy()
    dow = panel.times[origins].dayofweek.to_numpy()
    calendar = np.column_stack([np.sin(hour * np.pi / 12), np.cos(hour * np.pi / 12),
                                np.sin(dow * 2 * np.pi / 7), np.cos(dow * 2 * np.pi / 7)])
    pieces.append(np.broadcast_to(calendar[:, None], (t_count, n_count, 4)))
    coords = (panel.coords - panel.coords.mean(0)) * np.array([111.0, 89.0])
    pieces.append(np.broadcast_to(coords[None], (t_count, n_count, 2)))
    return np.concatenate(pieces, -1).astype(np.float32)


def split_origins(panel, history, horizon, train_end, valid_end):
    """A window is wholly within one partition; no boundary-crossing targets."""
    a, b = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    if not (panel.times[0] < a < b <= panel.times[-1]):
        raise ValueError("Require panel start < train_end < valid_end <= panel end.")
    result = {"train": [], "validation": [], "test": []}
    for t in range(history, len(panel.times) - horizon + 1):
        start, end = panel.times[t], panel.times[t] + pd.Timedelta(hours=horizon)
        if end <= a:
            part = "train"
        elif start >= a and end <= b:
            part = "validation"
        elif start >= b:
            part = "test"
        else:
            continue
        if np.isfinite(panel.counts[t:t + horizon]).any():
            result[part].append(t)
    if any(not v for v in result.values()):
        raise ValueError("Each partition needs at least one observed forecast window.")
    return {k: np.asarray(v, dtype=int) for k, v in result.items()}


def targets(panel, origins, horizon):
    # Model output is [B,N,H*2], channels alternate rentals and returns.
    y = np.stack([panel.counts[t:t + horizon].transpose(1, 0, 2).reshape(len(panel.stations), -1)
                  for t in origins])
    mask = np.isfinite(y)
    return np.nan_to_num(y, nan=0.0), mask


def distance_matrix(coords):
    rad = np.radians(coords)
    dlat = rad[:, None, 0] - rad[None, :, 0]
    dlon = rad[:, None, 1] - rad[None, :, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(rad[:, None, 0]) * np.cos(rad[None, :, 0]) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a.clip(0, 1)))


def build_graph(panel, train_end, neighbors=5, kind="hybrid"):
    """Only pre-cutoff observations can affect demand similarity edges."""
    n = len(panel.stations)
    if kind == "identity" or n == 1:
        return np.eye(n, dtype=np.float32)
    dist = distance_matrix(panel.coords)
    proximity = np.exp(-dist / 1.0)
    score = proximity
    if kind == "hybrid":
        before = panel.times < pd.Timestamp(train_end)
        profiles = np.zeros((n, 48), dtype=np.float32)
        for h in range(24):
            y = panel.counts[before & (panel.times.hour == h)]
            sums = np.nansum(y, axis=0)
            den = np.isfinite(y).sum(0)
            profiles[:, 2 * h:2 * h + 2] = sums / np.maximum(den, 1)
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        profiles = profiles / np.maximum(norms, 1e-6)
        score = 0.5 * proximity + 0.5 * (profiles @ profiles.T).clip(0, 1)
    elif kind != "distance":
        raise ValueError(f"Unknown graph: {kind}")
    np.fill_diagonal(score, 0)
    keep = np.argsort(-score, axis=1)[:, :min(neighbors, n - 1)]
    sparse = np.zeros_like(score)
    row = np.arange(n)[:, None]
    sparse[row, keep] = score[row, keep]
    sparse = np.maximum(sparse, sparse.T) + np.eye(n)
    return (sparse / sparse.sum(1, keepdims=True)).astype(np.float32)


def make_demo(path: Path, days=35, stations=12, seed=42):
    """Synthetic fleet with observed served trips. Never a claim about Tashu."""
    if days < 12 or stations < 2:
        raise ValueError("Demo needs at least 12 days and 2 stations.")
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    stock = rng.integers(7, 18, size=stations)
    coords = np.column_stack([36.35 + rng.normal(0, .012, stations), 127.38 + rng.normal(0, .015, stations)])
    destination = np.exp(-distance_matrix(coords) / 1.5) + .03
    destination /= destination.sum(1, keepdims=True)
    pending = np.zeros(stations, dtype=int)
    records = []
    for ts in times:
        morning = np.exp(-((ts.hour - 8) / 2) ** 2)
        evening = np.exp(-((ts.hour - 18) / 2.5) ** 2)
        scale = .1 + .6 * (7 <= ts.hour <= 22)
        rates = scale + 2.2 * morning * (np.arange(stations) % 2 == 0) + 2.2 * evening * (np.arange(stations) % 2 == 1)
        requested = rng.poisson(rates * (0.75 if ts.dayofweek >= 5 else 1.0))
        rentals = np.minimum(requested, stock)
        returns = pending.copy()
        for s in range(stations):
            records.append({"timestamp": ts.isoformat(), "station_id": f"DEMO{s:03d}",
                            "name": f"예시 대여소 {s + 1}", "rentals": int(rentals[s]), "returns": int(returns[s]),
                            "requested_rentals": int(requested[s]),
                            "stock": int(stock[s]), "lat": coords[s, 0], "lon": coords[s, 1]})
        stock = stock - rentals + returns
        pending = sum((rng.multinomial(int(rentals[s]), destination[s]) for s in range(stations)),
                      start=np.zeros(stations, dtype=int))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
    return times[int(.6 * len(times))], times[int(.8 * len(times))]
