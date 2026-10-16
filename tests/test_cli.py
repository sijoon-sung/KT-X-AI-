import json
import subprocess
import sys


def test_complete_demo_train_reload_compare_and_recommend(tmp_path):
    out = tmp_path / "demo"
    result = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "demo", "--out", str(out),
                             "--days", "12", "--stations", "4", "--history", "3", "--horizon", "2",
                             "--epochs", "1", "--hidden", "8", "--trees", "8", "--samples", "8",
                             "--ode-steps", "2", "--max-eval-origins", "8"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    evaluation = json.loads((out / "run/evaluation.json").read_text(encoding="utf-8"))
    recommendation = json.loads((out / "recommendation/recommendation.json").read_text(encoding="utf-8"))
    assert set(evaluation["models"]) == {"xgboost", "gcn", "gcn_flow"}
    assert evaluation["selection_metric"].startswith("validation_")
    assert recommendation["source"] == "synthetic"
    assert recommendation["initial_fleet_bikes"] == recommendation["final_fleet_bikes"]
    assert recommendation["moved_bikes"] <= recommendation["budget_bikes"]
    assert (out / "recommendation/report.html").exists()
    again = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "recommend", "--run", str(out / "run"),
                            "--panel", str(out / "synthetic_panel.csv"), "--model", "gcn_flow", "--samples", "8",
                            "--ode-steps", "2", "--out", str(tmp_path / "again")], capture_output=True, text=True)
    assert again.returncode == 0, again.stdout + again.stderr
    assert json.loads((tmp_path / "again/recommendation.json").read_text(encoding="utf-8"))["model"] == "gcn_flow"
