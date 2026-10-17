import numpy as np

from tashu_contest.audit import audit_panel
from tashu_contest.data import make_demo, make_features, read_panel


def test_audit_and_request_truth_never_enters_model_features(tmp_path):
    csv = tmp_path / "synthetic.csv"
    make_demo(csv, days=12, stations=3)
    panel = read_panel(csv)
    assert panel.requested_rentals is not None
    assert (panel.requested_rentals >= panel.counts[..., 0]).all()
    features = make_features(panel, 3, [4, 5])
    panel.requested_rentals[:] = 1000
    np.testing.assert_array_equal(features, make_features(panel, 3, [4, 5]))
    summary = audit_panel(csv, tmp_path / "audit")
    assert summary["count_observation_fraction"] == 1
    assert summary["stock_balance_mismatch_fraction"] == 0
    assert summary["source"] == "user_provided_unverified"
    assert (tmp_path / "audit/data_audit.json").exists()
