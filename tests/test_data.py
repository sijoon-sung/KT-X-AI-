import numpy as np
import pandas as pd
import pytest

from tashu_contest.data import build_graph, make_demo, make_features, read_panel, split_origins, targets


@pytest.fixture
def panel(tmp_path):
    path = tmp_path / "panel.csv"
    make_demo(path, days=12, stations=4)
    return read_panel(path)


def test_future_counts_and_stocks_cannot_change_origin_features(panel):
    at = 40
    old = make_features(panel, 24, [at]).copy()
    panel.counts[at:] = 9999
    panel.stock[at + 1:] = 9999
    np.testing.assert_array_equal(old, make_features(panel, 24, [at]))


def test_graph_cannot_use_validation_or_test_counts(panel):
    cutoff = panel.times[100]
    old = build_graph(panel, cutoff)
    panel.counts[100:] = np.arange(4)[None, :, None] * 999
    np.testing.assert_array_equal(old, build_graph(panel, cutoff))
    np.testing.assert_allclose(old.sum(1), 1, atol=1e-6)


def test_windows_do_not_cross_partition_boundaries(panel):
    parts = split_origins(panel, 24, 3, panel.times[140], panel.times[210])
    assert all(t + 3 <= 140 for t in parts["train"])
    assert all(140 <= t and t + 3 <= 210 for t in parts["validation"])
    assert all(t >= 210 for t in parts["test"])
    assert not (set(parts["train"]) & set(parts["validation"]))


def test_missing_observation_is_not_a_zero_label(panel):
    panel.counts[40, 0, 0] = np.nan
    panel.counts[40, 1, 0] = 0
    values, mask = targets(panel, [40], 1)
    assert not mask[0, 0, 0]
    assert mask[0, 1, 0] and values[0, 1, 0] == 0


def test_reader_rejects_duplicates_and_fractional_inventory(tmp_path):
    path = tmp_path / "panel.csv"
    make_demo(path, days=12, stations=2)
    frame = pd.read_csv(path, dtype={"station_id": str})
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_panel(path)
    frame["stock"] = frame.stock.astype(float)
    frame.loc[0, "stock"] = .5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonnegative integer"):
        read_panel(path)


def test_missing_hours_stay_missing_and_timezone_is_korean(tmp_path):
    path = tmp_path / "panel.csv"
    make_demo(path, days=12, stations=2)
    frame = pd.read_csv(path)
    frame = frame[frame.timestamp != frame.timestamp.iloc[4]]
    frame["timestamp"] = pd.to_datetime(frame.timestamp).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC").astype(str)
    frame.to_csv(path, index=False)
    result = read_panel(path)
    assert result.times[0] == pd.Timestamp("2026-01-01")
    assert np.isnan(result.counts[2]).all()
