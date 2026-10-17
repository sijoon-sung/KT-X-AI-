import numpy as np
import pandas as pd

from tashu_contest.backtest import operational_backtest
from tashu_contest.calibration import apply_calibration, fit_calibration
from tashu_contest.data import Panel


def panel_fixture():
    return Panel(pd.date_range("2026-01-01", periods=5, freq="h"), ["a", "b"],
                 np.zeros((5, 2, 2), dtype=np.float32),
                 np.array([[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]], dtype=np.float32),
                 np.array([[36., 127.], [36.001, 127.001]]), np.array([10., 10.]), ["a", "b"])


def test_validation_route_protects_counts_and_learns_separate_empty_recovery():
    panel = panel_fixture()
    actual = np.zeros((1, 2, 2, 2), dtype=np.float32)
    actual[:, :, 1, 1] = 1
    mask = np.ones_like(actual, dtype=bool)
    gcn = np.zeros((8, *actual.shape), dtype=np.float32)
    flow = np.broadcast_to(actual, gcn.shape).copy()
    flow[:, :, :, 0, 0] = 8  # Flow harms the currently positive station.
    calibration = fit_calibration(panel, np.array([0]), gcn, flow, actual, mask)
    assert calibration["fitted_on"] == "validation_only"
    assert calibration["in_stock"]["flow_weight"] == 0
    assert calibration["empty"]["flow_weight"] == 1
    result = apply_calibration(gcn, flow, panel.stock[[0]], calibration)
    np.testing.assert_array_equal(result, np.broadcast_to(actual, result.shape))
    assert np.isfinite(result).all() and (result == np.floor(result)).all()
    # The fitter sees only supplied validation origins, not future test labels.
    panel.stock[3:] = 0
    panel.counts[3:] = 100
    again = fit_calibration(panel, np.array([0]), gcn, flow, actual, mask)
    assert calibration == again


def test_replay_can_disagree_with_optimistic_model_scenarios():
    panel = panel_fixture()
    panel.counts[0, 0, 0] = 4  # Held-out rentals occur at donor, not receiver.
    scenarios = np.zeros((8, 1, 1, 2, 2), dtype=np.float32)
    scenarios[:, :, :, 1, 0] = 4
    report = operational_backtest(panel, np.array([0]), scenarios, maximum=1, budget=2)
    assert report["evaluated_origins"] == 1
    assert report["rows"][0]["model_expected_failures_reduction"] == 2
    assert report["policies"]["no_action"]["service_failures_mean"] == 0
    assert report["policies"]["marginal_benefit"]["failures_reduction_mean"] == -2
    assert report["policies"]["marginal_benefit"]["moved_bikes_mean"] == 2
    assert report["policies"]["nearest_zero_first"]["moved_bikes_mean"] == 2


def test_missing_city_counts_are_skipped_and_zero_limit_disables_replay():
    panel = panel_fixture()
    panel.counts[0, 1, 0] = np.nan
    values = np.zeros((8, 1, 1, 2, 2))
    assert operational_backtest(panel, np.array([0]), values)["evaluated_origins"] == 0
    panel.counts[0] = 0
    disabled = operational_backtest(panel, np.array([0]), values, maximum=0)
    assert disabled["status"] == "disabled"
    assert disabled["eligible_origins"] == 1


def test_requested_demand_is_held_out_of_planning_and_reports_separately():
    panel = panel_fixture()
    panel.requested_rentals = np.zeros((5, 2), dtype=np.float32)
    panel.requested_rentals[0, 1] = 4
    scenarios = np.zeros((8, 1, 1, 2, 2), dtype=np.float32)
    scenarios[:, :, :, 1, 0] = 4
    report = operational_backtest(panel, np.array([0]), scenarios, maximum=1, budget=2, source="synthetic")
    assert report["policies"]["no_action"]["service_failures_mean"] == 0
    demand = report["demand_replay"]
    assert demand["evidence"] == "synthetic_generator_requests"
    assert demand["policies"]["no_action"]["service_failures_mean"] == 4
    assert demand["policies"]["marginal_benefit"]["failures_reduction_mean"] == 2
    panel.requested_rentals[:] = 100
    changed = operational_backtest(panel, np.array([0]), scenarios, maximum=1, budget=2)
    assert changed["policies"]["marginal_benefit"]["moved_bikes_mean"] == 2
