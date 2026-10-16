import numpy as np

from tashu_contest.data import Panel
from tashu_contest.evaluation import stock_evidence, stock_metrics
import pandas as pd


def test_only_measured_stock_labels_are_evaluated_and_empty_recovery_is_separate():
    panel = Panel(pd.date_range("2026-01-01", periods=3, freq="h"), ["A", "B", "C"],
                  np.zeros((3, 3, 2), dtype=np.float32),
                  np.array([[1, 0, np.nan], [0, 1, np.nan], [0, 1, np.nan]]),
                  np.array([[36, 127], [36.01, 127], [36.02, 127]]),
                  np.full(3, np.nan), ["A", "B", "C"])
    samples = np.zeros((4, 2, 3, 2), dtype=int)
    samples[:, 0, 0, 0] = 1
    samples[:, 0, 1, 1] = 1
    metrics = stock_metrics([stock_evidence(panel, 0, samples)])
    assert metrics["cells"] == 1
    assert metrics["currently_empty_cells"] == 1
    assert metrics["brier"] == 0
    assert metrics["recovery_brier"] == 0
    assert metrics["auc"] is None


def test_stock_evaluation_is_explicitly_unavailable_without_measured_labels():
    assert stock_metrics([None])["status"] == "not_evaluated_no_measured_stock"
