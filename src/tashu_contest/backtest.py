"""Independent observed-transaction replay, never a causal lost-demand estimate."""
import numpy as np

from .data import distance_matrix
from .operations import plan_transfers, service_target, simulate


def nearest_topup(initial, scenarios, coords, capacity, budget, max_distance_km, service_level=.9):
    """Simple baseline: initially empty receivers first, then nearest surplus."""
    current = np.asarray(initial, dtype=int).copy()
    target = service_target(scenarios, service_level)
    target = np.minimum(target, np.where(np.isfinite(capacity), capacity, target)).astype(int)
    distances = distance_matrix(coords)
    moved = 0
    for receiver in sorted(range(len(initial)), key=lambda s: (initial[s] != 0, -max(target[s] - initial[s], 0), s)):
        for donor in np.argsort(distances[:, receiver], kind="stable"):
            if donor == receiver or distances[donor, receiver] > max_distance_km:
                continue
            amount = min(max(current[donor] - target[donor], 0), max(target[receiver] - current[receiver], 0), budget - moved)
            current[donor] -= amount
            current[receiver] += amount
            moved += int(amount)
            if moved == budget:
                return current, moved
    return current, moved


def replay_metrics(opening, observed, capacity, originally_empty):
    outcome = simulate(opening, observed[None], capacity)
    stock = outcome["stock"][0]
    rentals = float(outcome["unserved"].sum())
    returns = float(outcome["blocked_returns"].sum())
    return {"unserved_rentals": rentals, "blocked_returns": returns,
            "service_failures": rentals + returns,
            "stockout_station_hours": int((stock == 0).sum()),
            "initially_empty_recovered": int(((stock > 0).any(0) & originally_empty).sum())}


def operational_backtest(panel, origins, scenarios, maximum=8, budget=20, truck_capacity=10,
                         service_level=.9, max_distance_km=5., source="user_provided_unverified"):
    if maximum < 0:
        raise ValueError("Backtest origin limit must be nonnegative.")
    result = {"status": "disabled" if maximum == 0 else "no_complete_city_observations",
              "interpretation": "Fixed observed-transaction replay; censored demand and dispatch timing unobserved. Not a causal policy effect.",
              "eligible_origins": 0, "evaluated_origins": 0, "rows": [], "policies": {},
              "demand_replay": {"status": "not_available_no_requested_rentals", "evaluated_origins": 0, "policies": {}}}
    horizon = scenarios.shape[2]
    eligible = [i for i, origin in enumerate(origins)
                if np.isfinite(panel.stock[origin]).all() and
                np.isfinite(panel.counts[origin:origin + horizon]).all() and
                origin + horizon <= len(panel.times)]
    result["eligible_origins"] = len(eligible)
    if not maximum or not eligible:
        return result
    selected = np.asarray(eligible)[np.linspace(0, len(eligible) - 1, min(maximum, len(eligible)), dtype=int)]
    for i in selected:
        origin, forecast = int(origins[i]), scenarios[:, i]
        initial = panel.stock[origin].astype(int)
        # Plans only see opening stock + predictions; held-out transactions enter
        # after the plans have been fixed. Actual outcomes may get worse.
        plan = plan_transfers(initial, forecast, panel.coords, panel.capacity, budget,
                              truck_capacity, service_level, max_distance_km)
        simple, moved = nearest_topup(initial, forecast, panel.coords, panel.capacity, budget, max_distance_km, service_level)
        observed = panel.counts[origin:origin + horizon]
        policies, demand_policies = {}, {}
        requested = panel.requested_rentals[origin:origin + horizon] if panel.requested_rentals is not None else None
        demand = observed.copy()
        if requested is not None and np.isfinite(requested).all():
            demand[:, :, 0] = requested
        for name, stock, bikes in (("no_action", initial, 0), ("nearest_zero_first", simple, moved),
                                   ("marginal_benefit", plan["after_transfer"], plan["moved_bikes"])):
            policies[name] = {**replay_metrics(stock, observed, panel.capacity, initial == 0), "moved_bikes": bikes}
            if requested is not None and np.isfinite(requested).all():
                demand_policies[name] = {**replay_metrics(stock, demand, panel.capacity, initial == 0), "moved_bikes": bikes}
        result["rows"].append({"timestamp": str(panel.times[origin]), "policies": policies,
                               "demand_policies": demand_policies,
                               "model_expected_failures_reduction": plan["expected_failures_reduction"]})
    result["status"], result["evaluated_origins"] = "observed_transaction_replay", len(result["rows"])
    for name in ("no_action", "nearest_zero_first", "marginal_benefit"):
        records = [row["policies"][name] for row in result["rows"]]
        result["policies"][name] = {key + "_mean": float(np.mean([row[key] for row in records])) for key in records[0]}
        before = result["policies"]["no_action"]["service_failures_mean"]
        after = result["policies"][name]["service_failures_mean"]
        result["policies"][name]["failures_reduction_mean"] = before - after
    demand_rows = [row["demand_policies"] for row in result["rows"] if row["demand_policies"]]
    if demand_rows:
        replay = result["demand_replay"]
        replay.update(status="fixed_return_requested_demand_replay", evaluated_origins=len(demand_rows),
                      evidence="synthetic_generator_requests" if source == "synthetic" else "user_supplied_unverified_requests",
                      interpretation="Requested rentals are held out of planning. Returns remain fixed: not a full OD/causal simulation.")
        for name in ("no_action", "nearest_zero_first", "marginal_benefit"):
            replay["policies"][name] = {key + "_mean": float(np.mean([row[name][key] for row in demand_rows])) for key in demand_rows[0][name]}
            replay["policies"][name]["failures_reduction_mean"] = replay["policies"]["no_action"]["service_failures_mean"] - replay["policies"][name]["service_failures_mean"]
    return result
