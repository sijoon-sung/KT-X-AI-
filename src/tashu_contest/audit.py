"""Explicit evidence audit; never label legacy or generated inputs as real data."""
from pathlib import Path

import numpy as np

from .data import read_panel
from .pipeline import file_hash, write_json


def audit_panel(path, out):
    panel = read_panel(path)
    known_counts, known_stock = np.isfinite(panel.counts), np.isfinite(panel.stock)
    available = known_counts[:-1].all(2) & known_stock[:-1] & known_stock[1:]
    cap = np.where(np.isfinite(panel.capacity), panel.capacity, np.inf)
    expected = np.minimum(np.maximum(panel.stock[:-1] - panel.counts[:-1, :, 0], 0) + panel.counts[:-1, :, 1], cap)
    mismatch = np.abs(expected - panel.stock[1:])[available]
    summary = {"source": "user_provided_unverified", "panel_sha256": file_hash(path),
               "stations": len(panel.stations), "hours": len(panel.times),
               "first_timestamp": str(panel.times[0]), "last_timestamp": str(panel.times[-1]),
               "count_observation_fraction": float(known_counts.mean()),
               "opening_stock_observation_fraction": float(known_stock.mean()),
               "complete_city_stock_hours": int(known_stock.all(1).sum()),
               "known_capacity_stations": int(np.isfinite(panel.capacity).sum()),
               "stock_balance_checked_cells": int(len(mismatch)),
               "stock_balance_mismatch_fraction": float((mismatch > 0).mean()) if len(mismatch) else None,
               "stock_balance_mae": float(mismatch.mean()) if len(mismatch) else None,
               "interpretation": "Mismatch may reflect relocation, timestamps or intra-hour order; not automatic evidence of bad data. Provenance must be checked separately.",
               "warnings": []}
    if not known_stock.all(1).any():
        summary["warnings"].append("No complete city opening-stock snapshot: transfer replay/recommendation unavailable.")
    if not known_counts.all():
        summary["warnings"].append("Missing counts are masked for forecasts; incomplete city windows are skipped in operation replay.")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "data_audit.json", summary)
    print(f"Audit saved: {destination / 'data_audit.json'}; warnings={summary['warnings']}", flush=True)
    return summary
