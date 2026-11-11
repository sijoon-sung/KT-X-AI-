"""Capture the locked sampler's 16 integration steps for a synthetic example."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research_code'))
import rolling_inventory_model as model


def capture(weights):
    digest = hashlib.sha256((weights / 'model-lock.json').read_bytes()).hexdigest()
    if digest != '767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09':
        raise ValueError('The locked H13 checkpoint is required')
    model.configure()
    predictor = model.Predictor(weights, 'cpu')
    current = np.zeros(predictor.n)
    current[0] = 2
    valid = np.zeros(predictor.n, dtype=bool)
    valid[0] = True
    history = np.zeros((predictor.n, 4), dtype=np.float32)
    weather = np.zeros(54, dtype=np.float32)
    scheduled = model.h.stamp('2026-11-10 16:00')
    integrate = model.heun
    traces = []

    @torch.inference_mode()
    def record(flow, state, condition, center, steps=16):
        original = integrate(flow, state, condition, center, steps)
        path = [state.cpu().numpy().copy()]
        for step in range(steps):
            velocity = flow(state, state.new_tensor(step / steps), condition, center)
            trial = state + velocity / steps
            correction = flow(trial, state.new_tensor((step + 1) / steps), condition, center)
            state = state + (velocity + correction) / (2 * steps)
            path.append(state.cpu().numpy().copy())
        if not torch.equal(state, original):
            raise AssertionError('Trace differs from the locked integrator')
        traces.append(np.stack(path))
        return original

    with patch.object(model, 'heun', record):
        _, predictions = predictor.issue(scheduled, current, valid, history, weather)
    trajectory = np.concatenate(traces, axis=1)
    stock = predictions['recent_cfm'][0]
    if trajectory.shape != (17, 128, 2) or not np.isfinite(trajectory).all():
        raise AssertionError('Unexpected trace shape')
    values, counts = np.unique(stock, return_counts=True)
    return dict(model_lock_sha256=digest, input='one synthetic station condition; initial stock 2',
                integration_steps=16, samples=128, exact_original_integrator=True,
                coordinates=['rental log-count residual','return log-count residual'],
                trajectory=trajectory.tolist(), stock_histogram=dict(zip(map(int, values),map(int, counts))),
                weights_updated=False, inventory_api_requests=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.weights)
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'trajectory'}))
