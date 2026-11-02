"""Serving/packaging wrapper for locked H13 artifacts. No training commands."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / 'code').is_dir() else HERE.parent
CODE = ROOT / 'code' if (ROOT / 'code').is_dir() else ROOT / 'research_code'
sys.path.insert(0, str(CODE))
LOCK_SHA = '767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09'
ENDPOINT = 'https://bikeapp.tashu.or.kr:50041/v1/openapi/station'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def verify(folder):
    folder = Path(folder)
    if digest(folder / 'model-lock.json') != LOCK_SHA:
        raise ValueError('Original H13 model lock does not match.')
    lock = read(folder / 'model-lock.json')
    if lock.get('api_opened') is not False:
        raise ValueError('Historical-only model lock required.')
    for item in lock['artifacts']:
        name = item['id']
        if Path(name).name != name or digest(folder / name) != item['sha256']:
            raise ValueError('Model artifact mismatch: ' + name)
    package = ROOT / 'package-manifest.json'
    if package.exists():
        for item in read(package)['files']:
            p = (ROOT / item['path']).resolve()
            if not p.is_relative_to(ROOT.resolve()) or digest(p) != item['sha256']:
                raise ValueError('Package source mismatch.')
    return dict(model_lock_sha256=LOCK_SHA, artifacts=len(lock['artifacts']), verified=True)


def load_model(folder, device):
    verify(folder)
    import rolling_inventory_model as model
    import torch
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; choose --device cpu on the compute server.')
        model.configure()
    else:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
    return model.Predictor(folder, device)


def model_info(predictor):
    count = lambda obj: sum(p.numel() for p in obj.parameters())
    gcn = count(predictor.models['recent'][0])
    flow = count(predictor.flows[0])
    return dict(model='H13 GCN + conditional Flow Matching', model_lock_sha256=LOCK_SHA,
                gcn_parameters_per_seed=gcn, flow_parameters_per_seed=flow, seeds=[11, 29],
                fm_path_parameters=2 * (gcn + flow), all_loaded_parameters=4 * gcn + 2 * flow,
                trained_stations=predictor.n, horizon_seconds=600, particles=128, ode_steps=16,
                weather_dimension=54, weights_updated=False)


def as_snapshot(value, ids):
    import rolling_inventory_predict as original
    with tempfile.TemporaryDirectory(prefix='tashu-snapshot-') as tmp:
        p = Path(tmp) / 'snapshot.json'
        p.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return original.read_snapshot(p, ids)


def clock_values(offset, uncertainty):
    if offset is None or uncertainty is None or not math.isfinite(offset) or not math.isfinite(uncertainty) or not 0 <= uncertainty <= 2:
        raise ValueError('Live mode requires a verified --clock-offset and --clock-uncertainty (0..2 seconds).')
    return time.time() + offset


def predict(predictor, value, offset, uncertainty, replay=False):
    import numpy as np
    import rolling_inventory_predict as original
    import forecast_transfer_data as features
    if not isinstance(value, dict) or value.get('allow_missing_weather') is not True:
        raise ValueError('Set allow_missing_weather=true; this runner uses the original missing-weather encoding.')
    if set(value) - {'current', 'previous', 'allow_missing_weather'}:
        raise ValueError('Unknown input fields; future labels and extra traffic features are not accepted.')
    current = as_snapshot(value['current'], predictor.ids)
    previous = as_snapshot(value['previous'], predictor.ids) if value.get('previous') else None
    if not replay:
        now = clock_values(offset, uncertainty)
        if not current.scheduled <= now < current.scheduled + 600:
            raise ValueError('Current snapshot expired or is from the future.')
        stamps = current.actual[original.valid_observation(current)]
        if not len(stamps) or float(stamps.max()) > now + uncertainty:
            raise ValueError('Observation has not been received at issue time.')
    # Existing H13 weather encoder, with no observations: preserve age/missing flags.
    wx = features.weather([], [], [], 133, np.array([current.scheduled]))[0]
    forecast = original.issue(predictor, current, previous, wx)
    issued = time.time() + (offset or 0)
    if not replay and issued + uncertainty >= forecast['target_scheduled']:
        raise ValueError('Prediction target expired during inference.')
    result = dict(mode='synthetic_offline_demo' if replay else 'live_forecast',
                  model_lock_sha256=LOCK_SHA, weights_updated=False, weather_mode='missing',
                  scheduled_at_unix_seconds=current.scheduled,
                  target_at_unix_seconds=forecast['target_scheduled'],
                  issued_at_utc=datetime.fromtimestamp(issued, timezone.utc).isoformat(),
                  supported_valid_stations=len(forecast['nodes']), stations=[])
    for j, node in enumerate(forecast['nodes']):
        item = dict(station_id=str(predictor.ids[node]), current_stock=int(forecast['current'][j]), models={})
        for label, samples in forecast['draws'].items():
            x = samples[j]
            item['models'][label] = dict(mean=float(x.mean()), median=float(np.sort(x)[63]),
                                        probability_zero=float((x == 0).mean()),
                                        q05=float(np.quantile(x, .05)), q95=float(np.quantile(x, .95)))
        result['stations'].append(item)
    return result


def demo_input(predictor):
    scheduled = int(datetime(2026, 11, 2, 9, tzinfo=timezone.utc).timestamp())
    rows = [dict(id=str(s), parking_count=i % 5) for i, s in enumerate(predictor.ids[:5])]
    return dict(current=dict(scheduled_at_unix_seconds=scheduled,
                             observed_at_unix_seconds=scheduled + 2, payload=dict(results=rows)),
                allow_missing_weather=True)


def serve(predictor, port, offset, uncertainty):
    clock_values(offset, uncertainty)
    mutex = threading.Lock()
    class API(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send(self, status, body):
            raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            if self.path == '/health':
                self.send(200, dict(status='ready', model_lock_sha256=LOCK_SHA))
            elif self.path == '/api/model':
                self.send(200, model_info(predictor))
            else:
                self.send(404, dict(error='Unknown route.'))
        def do_POST(self):
            if self.path != '/api/predict':
                return self.send(404, dict(error='Unknown route.'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4_000_000:
                    raise ValueError('Request must contain 1..4000000 bytes.')
                value = json.loads(self.rfile.read(size))
                with mutex:
                    result = predict(predictor, value, offset, uncertainty)
                self.send(200, result)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                self.send(422, dict(error='Invalid, expired, or unsupported forecast input.'))
            except Exception:
                self.send(503, dict(error='Prediction unavailable.'))
    server = ThreadingHTTPServer(('127.0.0.1', port), API)
    print(f'Local: http://127.0.0.1:{port}/health', flush=True)
    server.serve_forever()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def capture(a):
    if Path(a.out).exists():
        raise ValueError('Output exists; choose a new filename.')
    token = os.environ.get('TASHU_API_KEY', '')
    if a.credential:
        import live_api_service
        token = live_api_service.credential(a.credential)
    if not token:
        raise ValueError('Set TASHU_API_KEY or pass --credential to the existing encrypted file.')
    now = clock_values(a.clock_offset, a.clock_uncertainty)
    scheduled = math.floor(now / 600) * 600
    if now - scheduled > 90:
        if not a.wait:
            raise ValueError('Capture within 90 seconds after a 10-minute boundary, or use --wait.')
        scheduled += 600
        while clock_values(a.clock_offset, a.clock_uncertainty) < scheduled:
            time.sleep(max(0, min(15, scheduled - clock_values(a.clock_offset, a.clock_uncertainty))))
    request = Request(ENDPOINT, headers={'api-token': token, 'User-Agent': 'TashuLockedRuntime/1.0'})
    try:
        with build_opener(NoRedirect).open(request, timeout=15) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError('API response too large.')
        payload = json.loads(raw)
    except Exception:
        raise ValueError('API request failed; credential and response were not logged.') from None
    observed = clock_values(a.clock_offset, a.clock_uncertainty)
    rows = payload.get('results') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows or observed - scheduled > 120:
        raise ValueError('Missing or delayed station response.')
    ids = [str(r['id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate station IDs.')
    write_new(a.out, dict(scheduled_at_unix_seconds=scheduled,
                          observed_at_unix_seconds=observed, payload=payload))
    print(json.dumps(dict(saved=True, station_rows=len(rows), api_requests=1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'inspect', 'demo', 'predict', 'serve', 'capture'])
    p.add_argument('--model-dir', default=str(ROOT / 'model'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out')
    p.add_argument('--current')
    p.add_argument('--previous')
    p.add_argument('--allow-missing-weather', action='store_true')
    p.add_argument('--clock-offset', type=float)
    p.add_argument('--clock-uncertainty', type=float)
    p.add_argument('--port', type=int, default=8780)
    p.add_argument('--credential')
    p.add_argument('--wait', action='store_true')
    a = p.parse_args()
    if a.command == 'capture':
        if not a.out:
            p.error('capture requires --out')
        return capture(a)
    if a.command == 'verify':
        print(json.dumps(verify(a.model_dir)))
        return
    predictor = load_model(a.model_dir, a.device)
    if a.command == 'inspect':
        print(json.dumps(model_info(predictor)))
    elif a.command == 'serve':
        serve(predictor, a.port, a.clock_offset, a.clock_uncertainty)
    else:
        if a.out and Path(a.out).exists():
            raise ValueError('Output exists; choose a new filename.')
        if a.command == 'demo':
            value = demo_input(predictor)
        else:
            if not a.current:
                p.error('predict requires --current')
            value = dict(current=read(a.current), allow_missing_weather=a.allow_missing_weather)
            if a.previous:
                value['previous'] = read(a.previous)
        result = predict(predictor, value, a.clock_offset, a.clock_uncertainty, replay=a.command == 'demo')
        if a.out:
            write_new(a.out, result)
            print(json.dumps(dict(saved=True, mode=result['mode'], stations=result['supported_valid_stations'])))
        else:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
