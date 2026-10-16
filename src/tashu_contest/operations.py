"""Integer inventory simulation and feasible direct-transfer batches.

Hourly convention: rentals precede returns (conservative within-hour ordering).
Recommendations minimize a model-based service-failure proxy, not measured lost
demand. Demand scenarios remain fixed across before/after plan comparisons.
"""
import numpy as np

from .data import distance_matrix


def simulate(initial, scenarios, capacity=None):
    """Scenarios [S,H,N,2]; no negative inventory or fractional bikes."""
    initial = np.asarray(initial, dtype=float)
    if not np.isfinite(initial).all() or (initial < 0).any() or (initial != np.floor(initial)).any():
        raise ValueError("Simulation needs measured nonnegative integer opening stocks.")
    values = np.asarray(scenarios)
    if values.ndim != 4 or values.shape[-1] != 2 or values.shape[2] != len(initial):
        raise ValueError("Expected scenarios [samples,horizon,stations,2].")
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Scenario counts must be finite nonnegative integers.")
    cap = np.full(len(initial), np.inf) if capacity is None else np.where(np.isfinite(capacity), capacity, np.inf)
    if (initial > cap).any():
        raise ValueError("Opening stock exceeds capacity.")
    stock = np.broadcast_to(initial, (len(values), len(initial))).copy()
    paths, lost, blocked = [], np.zeros_like(stock), np.zeros_like(stock)
    for hour in range(values.shape[1]):
        rentals, returns = values[:, hour, :, 0], values[:, hour, :, 1]
        served = np.minimum(stock, rentals)
        lost += rentals - served
        stock -= served
        accepted = np.minimum(returns, np.maximum(cap - stock, 0))
        blocked += returns - accepted
        stock += accepted
        paths.append(stock.copy())
    return {"stock": np.stack(paths, axis=1), "unserved": lost, "blocked_returns": blocked}


def service_target(scenarios, quantile=.9):
    rents, rets = scenarios[..., 0], scenarios[..., 1]
    before_returns = np.cumsum(rets, axis=1) - rets
    required = np.maximum(np.max(np.cumsum(rents, axis=1) - before_returns, axis=1), 0)
    return np.quantile(required, quantile, axis=0, method="higher").astype(int)


def plan_transfers(initial, scenarios, coords, capacity=None, budget=20, truck_capacity=10,
                   service_level=.9, max_distance_km=5., distance_weight=.1):
    """Greedy positive marginal net benefit; direct batches, not a VRP route."""
    if budget < 0 or truck_capacity < 1 or not 0 < service_level < 1 or max_distance_km <= 0:
        raise ValueError("Invalid movement budget, truck capacity, service level or distance.")
    before = simulate(initial, scenarios, capacity)
    initial = np.asarray(initial, dtype=int)
    target = service_target(scenarios, service_level)
    if capacity is not None:
        target = np.minimum(target, np.where(np.isfinite(capacity), capacity, target)).astype(int)
    current = initial.copy()
    distances = distance_matrix(coords)
    n = len(initial)
    losses = {}

    def loss(station, bikes):
        key = (station, int(bikes))
        if key not in losses:
            cap = None if capacity is None else np.asarray(capacity)[station:station + 1]
            result = simulate([bikes], scenarios[:, :, station:station + 1], cap)
            losses[key] = float((result["unserved"] + result["blocked_returns"]).mean())
        return losses[key]

    grouped = {}
    # Receiver gains and donor costs are separable, so only touched entries need updating.
    gain, cost = np.full(n, -np.inf), np.full(n, np.inf)

    def update(s):
        gain[s] = loss(s, current[s]) - loss(s, current[s] + 1) if current[s] < target[s] else -np.inf
        cost[s] = loss(s, current[s] - 1) - loss(s, current[s]) if current[s] > target[s] else np.inf

    for s in range(n):
        update(s)
    for _ in range(budget):
        benefit = gain[None, :] - cost[:, None]
        score = benefit / (1 + distance_weight * distances)
        score[(distances > max_distance_km) | np.eye(n, dtype=bool)] = -np.inf
        flat = int(np.argmax(score))
        donor, receiver = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[donor, receiver]) or benefit[donor, receiver] <= 1e-8:
            break
        current[donor] -= 1
        current[receiver] += 1
        grouped[(int(donor), int(receiver))] = grouped.get((int(donor), int(receiver)), 0) + 1
        update(donor)
        update(receiver)
    transfers = []
    for (donor, receiver), amount in grouped.items():
        while amount:
            take = min(amount, truck_capacity)
            transfers.append({"batch": len(transfers) + 1, "from_index": donor, "to_index": receiver,
                              "bikes": take, "direct_distance_km": float(distances[donor, receiver])})
            amount -= take
    after = simulate(current, scenarios, capacity)
    moved = int(sum(r["bikes"] for r in transfers))
    if current.sum() != initial.sum() or moved > budget or (current < 0).any():
        raise AssertionError("Transfer plan violated fleet/budget constraints.")
    loss_before = float((before["unserved"] + before["blocked_returns"]).sum(1).mean())
    loss_after = float((after["unserved"] + after["blocked_returns"]).sum(1).mean())
    return {"target": target, "initial": initial, "after_transfer": current, "transfers": transfers,
            "moved_bikes": moved, "requested_topup_bikes": int(np.maximum(target - initial, 0).sum()),
            "unfilled_topup_bikes": int(np.maximum(target - current, 0).sum()),
            "service_failures_before": loss_before, "service_failures_after": loss_after,
            "expected_failures_reduction": loss_before - loss_after,
            "stock_before": before["stock"], "stock_after": after["stock"]}


def station_summary(panel, initial, scenarios, plan):
    paths = plan["stock_after"]
    rows = []
    for s, sid in enumerate(panel.stations):
        terminal = paths[:, :, s]
        rows.append({"station_id": sid, "name": panel.names[s], "lat": float(panel.coords[s, 0]),
                     "lon": float(panel.coords[s, 1]), "current_stock": int(initial[s]),
                     "currently_empty": bool(initial[s] == 0), "target_stock": int(plan["target"][s]),
                     "requested_topup": int(max(plan["target"][s] - initial[s], 0)),
                     "planned_stock": int(plan["after_transfer"][s]),
                     "empty_probability_before": float((plan["stock_before"][:, :, s] == 0).any(1).mean()),
                     "empty_probability_after": float((terminal == 0).any(1).mean()),
                     "recovery_probability_after": float((terminal > 0).any(1).mean()) if initial[s] == 0 else None,
                     "forecast_stock_p10": np.quantile(terminal, .1, axis=0).tolist(),
                     "forecast_stock_p50": np.quantile(terminal, .5, axis=0).tolist(),
                     "forecast_stock_p90": np.quantile(terminal, .9, axis=0).tolist(),
                     "forecast_rentals_mean": scenarios[:, :, s, 0].mean(0).tolist(),
                     "forecast_returns_mean": scenarios[:, :, s, 1].mean(0).tolist()})
    return sorted(rows, key=lambda r: (-r["empty_probability_before"], -r["requested_topup"]))
