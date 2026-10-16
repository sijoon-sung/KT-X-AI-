import numpy as np
import pytest

from tashu_contest.operations import plan_transfers, simulate


def test_inventory_accounting_nonnegative_and_capacity_over_many_scenarios():
    rng = np.random.default_rng(3)
    values = rng.poisson(3, (100, 5, 6, 2))
    initial = np.arange(6)
    capacity = np.full(6, 8)
    out = simulate(initial, values, capacity)
    assert out["stock"].min() >= 0
    assert out["stock"].max() <= 8
    assert np.equal(out["stock"], np.floor(out["stock"])).all()
    served = values[..., 0].sum(1) - out["unserved"]
    accepted = values[..., 1].sum(1) - out["blocked_returns"]
    np.testing.assert_array_equal(out["stock"][:, -1], initial - served + accepted)


def test_transfer_has_concrete_bike_counts_and_improves_same_scenarios():
    demand = np.zeros((20, 2, 3, 2), dtype=int)
    demand[:, 0, 1, 0] = 4
    initial = np.array([10, 0, 0])
    coords = np.array([[36.35, 127.38], [36.351, 127.381], [36.352, 127.382]])
    plan = plan_transfers(initial, demand, coords, budget=3, truck_capacity=2)
    assert plan["moved_bikes"] == 3
    np.testing.assert_array_equal(plan["after_transfer"], [7, 3, 0])
    assert all(0 < transfer["bikes"] <= 2 for transfer in plan["transfers"])
    assert plan["after_transfer"].sum() == initial.sum()
    assert plan["expected_failures_reduction"] == 3
    assert plan["unfilled_topup_bikes"] == 1


def test_cannot_create_bikes_when_city_has_no_donors():
    demand = np.ones((10, 2, 2, 2), dtype=int)
    plan = plan_transfers([0, 0], demand, np.array([[36., 127.], [36.001, 127.001]]), budget=100)
    assert not plan["transfers"]
    assert plan["after_transfer"].sum() == 0
    assert plan["unfilled_topup_bikes"] > 0


def test_distance_and_zero_budget_are_hard_constraints():
    demand = np.zeros((5, 1, 2, 2), dtype=int)
    demand[:, :, 1, 0] = 4
    coords = np.array([[36., 127.], [37., 128.]])
    assert not plan_transfers([10, 0], demand, coords, max_distance_km=1)["transfers"]
    assert not plan_transfers([10, 0], demand, coords, budget=0)["transfers"]


def test_missing_inventory_must_not_silently_become_zero():
    with pytest.raises(ValueError, match="measured"):
        simulate([np.nan], np.zeros((2, 1, 1, 2)))
