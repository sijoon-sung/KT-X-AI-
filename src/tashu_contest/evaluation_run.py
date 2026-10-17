"""Evaluate existing artifacts, fit validation calibration, then replay test policies."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import operational_backtest
from .calibration import apply_calibration, fit_calibration
from .data import make_features, read_panel, targets
from .evaluation import count_metrics, stock_evidence, stock_metrics
from .models import load_models, predict_scenarios, seed_everything
from .pipeline import MODELS, device_for, file_hash, sync, write_json


def evaluate_saved(args):
    run = Path(args.run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    panel = read_panel(args.panel)
    if file_hash(args.panel) != metadata["panel_sha256"]:
        raise ValueError("Evaluation requires the identical training panel CSV (hash mismatch). Use train for a new dataset.")
    device = device_for(args.device or "cpu")
    for key, default in (("samples", 32), ("ode_steps", 8), ("sample_chunk", 4), ("eval_batch_size", 2)):
        if getattr(args, key, None) is None:
            setattr(args, key, metadata.get(key, default))
    if args.samples < 2 or min(args.ode_steps, args.sample_chunk, args.eval_batch_size) < 1 or args.backtest_origins < 0:
        raise ValueError("Invalid evaluation samples, steps, batches or backtest limit.")
    if args.budget < 0 or args.truck_capacity < 1 or not 0 < args.service_level < 1 or args.max_distance_km <= 0:
        raise ValueError("Invalid backtest movement constraints.")
    seed_everything(metadata["seed"])
    models = load_models(run / "models", metadata, device)
    graph = np.load(run / "graph.npy", allow_pickle=False)
    cache = run / "evaluation_cache"
    cache.mkdir(exist_ok=True)
    origins_by_part, actual_by_part, masks_by_part, samples_by_part, timings = {}, {}, {}, {}, {}
    for part in ("validation", "test"):
        origins = panel.times.get_indexer(pd.to_datetime(metadata["evaluated_origins"][part]))
        if (origins < metadata["history"]).any() or (origins + metadata["horizon"] > len(panel.times)).any():
            raise ValueError("Stored evaluation origins are incompatible with panel history/horizon.")
        origins_by_part[part] = origins
        features = make_features(panel, metadata["history"], origins)
        y, mask = targets(panel, origins, metadata["horizon"])
        actual_by_part[part] = y.reshape(len(origins), len(panel.stations), metadata["horizon"], 2).transpose(0, 2, 1, 3)
        masks_by_part[part] = mask.reshape(len(origins), len(panel.stations), metadata["horizon"], 2).transpose(0, 2, 1, 3)
        samples_by_part[part], timings[part] = {}, {}
        for name in MODELS:
            shape = (args.samples, len(origins), metadata["horizon"], len(panel.stations), 2)
            values = np.lib.format.open_memmap(cache / f"{part}_{name}.npy", mode="w+", dtype=np.float32, shape=shape)
            predict_scenarios(name, models, features[:1], graph, metadata["horizon"], args.samples,
                              args.ode_steps, metadata["seed"] + 1999, args.sample_chunk, args.amp)
            sync(device)
            seconds = 0.
            for offset in range(0, len(origins), args.eval_batch_size):
                sync(device)
                start = time.perf_counter()
                batch = predict_scenarios(name, models, features[offset:offset + args.eval_batch_size], graph,
                                          metadata["horizon"], args.samples, args.ode_steps,
                                          metadata["seed"] + 2000 + offset, args.sample_chunk, args.amp)
                sync(device)
                seconds += time.perf_counter() - start
                values[:, offset:offset + len(batch[0])] = batch
            values.flush()
            samples_by_part[part][name] = values
            timings[part][name] = seconds
            print(f"Evaluated {part}/{name}: {len(origins)} origins", flush=True)
    val = samples_by_part["validation"]
    calibration = fit_calibration(panel, origins_by_part["validation"], val["gcn"], val["gcn_flow"],
                                  actual_by_part["validation"], masks_by_part["validation"])
    calibration["scenario_settings"] = {"samples": args.samples, "ode_steps": args.ode_steps,
                                        "sample_chunk": args.sample_chunk, "seed": metadata["seed"]}
    results, operations = {name: {} for name in (*MODELS, "gcn_flow_calibrated")}, {}
    for part in ("validation", "test"):
        origins, samples = origins_by_part[part], samples_by_part[part]
        start = time.perf_counter()
        samples["gcn_flow_calibrated"] = apply_calibration(samples["gcn"], samples["gcn_flow"], panel.stock[origins], calibration)
        blend_seconds = time.perf_counter() - start
        uses_flow = any(calibration[g]["flow_weight"] > 0 for g in ("in_stock", "empty"))
        timings[part]["gcn_flow_calibrated"] = blend_seconds + timings[part]["gcn"] + (timings[part]["gcn_flow"] if uses_flow else 0)
        for name, values in samples.items():
            evidence = [stock_evidence(panel, origin, values[:, i]) for i, origin in enumerate(origins)]
            seconds = timings[part][name]
            results[name][part] = {"counts": count_metrics(values, actual_by_part[part], masks_by_part[part]),
                                   "stock": stock_metrics(evidence),
                                   "inference": {"seconds": seconds, "ms_per_city_origin": seconds * 1000 / len(origins),
                                                 "samples_per_origin": args.samples, "batch_origins": args.eval_batch_size,
                                                 "scenario_chunk": args.sample_chunk,
                                                 "amp": bool(name != "xgboost" and args.amp and device.type == "cuda"),
                                                 "device": "cpu" if name == "xgboost" else str(device),
                                                 "timing_method": "shared forecasts + mixture application" if name == "gcn_flow_calibrated" else "measured"}}
            if part == "test":
                operations[name] = operational_backtest(panel, origins, values, args.backtest_origins, args.budget,
                                                        args.truck_capacity, args.service_level, args.max_distance_km, metadata["source"])
    # Equal normalized objectives: don't pick a stockout win that discards counts
    # or already-empty recovery quality. Test outcomes never choose this winner.
    objectives = [("counts", "mae"), ("stock", "net_inventory_mae"), ("stock", "brier"), ("stock", "recovery_brier")]
    objectives = [(section, key) for section, key in objectives
                  if all(results[name]["validation"][section].get(key) is not None for name in results)]
    scores = {name: float(np.mean([results[name]["validation"][section][key] /
                                  max(results["gcn"]["validation"][section][key], 1e-6)
                                  for section, key in objectives])) for name in results}
    best = min(scores, key=scores.get)
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        code_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        code_commit, code_dirty = None, None
    report = {"source": metadata["source"], "selection_metric": "validation_balanced_count_stock_stockout_recovery",
              "selection_scores": scores, "selection_objectives": [f"{s}.{k}" for s, k in objectives],
              "selected_model": best,
              "models": results, "calibration": calibration,
              "operations": {"source": metadata["source"], "partition": "test", "budget_bikes": args.budget,
                             "truck_capacity": args.truck_capacity, "service_level": args.service_level,
                             "max_distance_km": args.max_distance_km, "models": operations},
              "evaluation_settings": {"code_panel_hash_verified": True, "code_commit": code_commit,
                                      "code_dirty": code_dirty, "seed": metadata["seed"],
                                      "samples": args.samples, "ode_steps": args.ode_steps,
                                      "sample_chunk": args.sample_chunk,
                                      "backtest_origins": args.backtest_origins},
              "limitations": metadata["limitations"] + ["Calibration fitted on validation only; selected on that same validation set. Single seed; no uncertainty interval for score differences.",
                                                        "Replay holds observed transactions fixed; not causal avoided stockouts or uncensored demand."]}
    if (run / "evaluation.json").exists() and not (run / "evaluation_before_upgrade.json").exists():
        shutil.copy2(run / "evaluation.json", run / "evaluation_before_upgrade.json")
    write_json(run / "calibration.json", calibration)
    write_json(run / "evaluation.json", report)
    write_json(run / "operations_evaluation.json", report["operations"])
    print(f"Evaluation saved: {run / 'evaluation.json'}; selected={best}", flush=True)
    return run
