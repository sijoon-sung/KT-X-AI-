import json
import subprocess
import sys
import csv
import pytest


@pytest.mark.parametrize("architecture", [[], ["--gcn-layers", "4", "--flow-blocks", "3", "--dropout", ".1",
                                             "--accumulate", "2", "--sample-chunk", "2"]], ids=["base", "deep"])
def test_complete_demo_train_reload_compare_and_recommend(tmp_path, architecture):
    out = tmp_path / "demo"
    result = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "demo", "--out", str(out),
                             "--days", "12", "--stations", "4", "--history", "3", "--horizon", "2",
                             "--epochs", "1", "--hidden", "8", "--trees", "8", "--samples", "8",
                             "--ode-steps", "2", "--max-eval-origins", "8"] + architecture, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    evaluation = json.loads((out / "run/evaluation.json").read_text(encoding="utf-8"))
    metadata = json.loads((out / "run/metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == "custom"
    assert metadata["parameter_counts"]["gcn_flow"]["frozen"] > 0
    assert metadata["training_seconds"]["gcn_flow_including_base"] > 0
    recommendation = json.loads((out / "recommendation/recommendation.json").read_text(encoding="utf-8"))
    assert set(evaluation["models"]) == {"xgboost", "gcn", "gcn_flow", "gcn_flow_calibrated"}
    assert evaluation["calibration"]["fitted_on"] == "validation_only"
    assert evaluation["operations"]["partition"] == "test"
    assert evaluation["operations"]["models"]["gcn_flow_calibrated"]["evaluated_origins"] > 0
    assert (out / "run/calibration.json").exists()
    assert (out / "run/operations_evaluation.json").exists()
    assert evaluation["models"]["gcn_flow"]["test"]["inference"]["ms_per_city_origin"] > 0
    assert evaluation["selection_metric"].startswith("validation_")
    assert recommendation["source"] == "synthetic"
    assert recommendation["initial_fleet_bikes"] == recommendation["final_fleet_bikes"]
    assert recommendation["moved_bikes"] <= recommendation["budget_bikes"]
    assert (out / "recommendation/report.html").exists()
    with (out / "recommendation/hourly_forecasts.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 2
    assert {row["hours_ahead"] for row in rows} == {"1", "2"}
    again = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "recommend", "--run", str(out / "run"),
                            "--panel", str(out / "synthetic_panel.csv"), "--model", "gcn_flow", "--samples", "8",
                            "--ode-steps", "2", "--out", str(tmp_path / "again")], capture_output=True, text=True)
    assert again.returncode == 0, again.stdout + again.stderr
    reloaded = json.loads((tmp_path / "again/recommendation.json").read_text(encoding="utf-8"))
    assert reloaded["model"] == "gcn_flow"
    assert reloaded["sample_chunk"] == metadata["sample_chunk"]
    assert metadata["flow_blocks"] == (3 if architecture else 1)
    calibrated = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "recommend", "--run", str(out / "run"),
                                "--panel", str(out / "synthetic_panel.csv"), "--model", "gcn_flow_calibrated",
                                "--out", str(tmp_path / "calibrated")], capture_output=True, text=True)
    assert calibrated.returncode == 0, calibrated.stdout + calibrated.stderr
    if not architecture:
        rerun = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "evaluate", "--run", str(out / "run"),
                               "--panel", str(out / "synthetic_panel.csv"), "--backtest-origins", "2"], capture_output=True, text=True)
        assert rerun.returncode == 0, rerun.stdout + rerun.stderr
        assert (out / "run/evaluation_before_upgrade.json").exists()
        altered = tmp_path / "altered.csv"
        altered.write_bytes((out / "synthetic_panel.csv").read_bytes() + b"\n")
        rejected = subprocess.run([sys.executable, "-m", "tashu_contest.cli", "evaluate", "--run", str(out / "run"),
                                  "--panel", str(altered)], capture_output=True, text=True)
        assert rejected.returncode == 2 and "hash mismatch" in rejected.stderr


def test_profiles_preserve_base_and_allow_explicit_override(monkeypatch):
    from tashu_contest import cli, pipeline
    seen = []
    monkeypatch.setattr(pipeline, "fit", lambda args: seen.append(args.hidden))
    common = ["train", "--panel", "unused.csv", "--train-end", "2026-02-01", "--valid-end", "2026-03-01", "--out", "unused"]
    cli.main(common)
    cli.main(common + ["--profile", "medium"])
    cli.main(common + ["--profile", "medium", "--hidden", "64"])
    assert seen == [48, 96, 64]


def test_rtx_profile_resolves_cuda_depth_and_memory_defaults(monkeypatch):
    from tashu_contest import cli, pipeline
    seen = []
    monkeypatch.setattr(pipeline, "fit", lambda args: seen.append(args))
    cli.main(["train", "--profile", "rtx3080", "--panel", "unused.csv", "--train-end", "2026-02-01",
              "--valid-end", "2026-03-01", "--out", "unused"])
    args = seen[0]
    assert (args.hidden, args.gcn_layers, args.flow_blocks) == (512, 4, 6)
    assert (args.device, args.xgb_device, args.epochs) == ("cuda", "cuda", 60)
    assert args.batch_size * args.accumulate == 16 and args.sample_chunk == 4
