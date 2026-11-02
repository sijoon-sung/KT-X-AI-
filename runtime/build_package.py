"""Package original locked artifacts on dislab; never reads API observations."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = '767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09'
ALLOWED = {'context.npz', 'time-features.npz', 'flow-11.pt', 'flow-29.pt',
           'recent-11.pt', 'recent-29.pt', 'no_recent-11.pt', 'no_recent-29.pt'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fit-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    fit, out = Path(a.fit_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    bundle = out / 'tashu-h13-runtime-v1'
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'code').mkdir()
    lock_path = fit / 'model-lock.json'
    assert digest(lock_path) == LOCK_SHA
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert lock['api_opened'] is False
    assert {x['id'] for x in lock['artifacts']} == ALLOWED
    shutil.copyfile(lock_path, bundle / 'model/model-lock.json')
    for item in lock['artifacts']:
        original = fit / item['id']
        assert digest(original) == item['sha256'], item['id']
        shutil.copyfile(original, bundle / 'model' / item['id'])
    # Copy only transitive Python module dependencies, with canonical source bytes.
    queue = ['rolling_inventory_model', 'rolling_inventory_predict', 'live_api_service', 'forecast_transfer_data']
    selected = set()
    original_code = {x['id']: x['sha256'] for x in lock['code']}
    while queue:
        module = queue.pop()
        if module in selected:
            continue
        selected.add(module)
        path = ROOT / 'research_code' / (module + '.py')
        raw = path.read_bytes().replace(b'\r\n', b'\n')
        if path.name in original_code:
            assert hashlib.sha256(raw).hexdigest() == original_code[path.name], ('Frozen source differs', path.name)
        (bundle / 'code' / path.name).write_bytes(raw)
        for node in ast.walk(ast.parse(raw.decode('utf-8'))):
            names = [x.name.split('.')[0] for x in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else []
            queue += [x for x in names if (ROOT / 'research_code' / (x + '.py')).is_file()]
    shutil.copyfile(ROOT / 'runtime/run.py', bundle / 'run.py')
    shutil.copyfile(ROOT / 'runtime/requirements.txt', bundle / 'requirements.txt')
    shutil.copyfile(ROOT / 'docs/quickstart.md', bundle / 'README.md')
    shutil.copyfile(ROOT / 'docs/model-card-h13.md', bundle / 'MODEL_CARD.md')
    shutil.copyfile(ROOT / 'docs/dispatch-runbook.md', bundle / 'dispatch-runbook.md')
    sys.path.insert(0, str(bundle / 'code'))
    import rolling_inventory_model as m
    import torch
    gcn = m.Forecaster()
    flow = m.CountFlow(context_width=256, width=128)
    count = lambda obj: sum(v.numel() for v in obj.parameters())
    weights = []
    for file in sorted((bundle / 'model').glob('*.pt')):
        state = torch.load(file, map_location='cpu', weights_only=True)['state_dict']
        model = m.CountFlow(context_width=256, width=128) if file.name.startswith('flow') else m.Forecaster()
        model.load_state_dict(state, strict=True)
        weights.append(dict(name=file.name, parameters=count(model), file_bytes=file.stat().st_size,
                            tensor_bytes=sum(v.numel() * v.element_size() for v in state.values()), sha256=digest(file)))
    report = dict(model_id='H13-locked-v1', model_lock_sha256=LOCK_SHA,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                  gcn_parameters_per_seed=count(gcn), flow_parameters_per_seed=count(flow),
                  encoder_parameters_per_seed=count(gcn.encoder),
                  fm_path_parameters=2 * (count(gcn) + count(flow)),
                  all_loaded_parameters=4 * count(gcn) + 2 * count(flow),
                  checkpoint_file_bytes=sum(x['file_bytes'] for x in weights),
                  checkpoint_tensor_bytes=sum(x['tensor_bytes'] for x in weights),
                  context_file_bytes=(bundle / 'model/context.npz').stat().st_size,
                  runtime_models=weights, code_modules=sorted(selected),
                  environment={k: importlib.metadata.version(k) for k in ['torch', 'numpy', 'scipy', 'pandas']},
                  raw_api_included=False, raw_trip_events_included=False, credentials_included=False,
                  model_retrained=False, scope='packaging existing historical-only fixed artifacts')
    write(bundle / 'model-info.json', report)
    files = [dict(path=f.relative_to(bundle).as_posix(), bytes=f.stat().st_size, sha256=digest(f))
             for f in sorted(bundle.rglob('*')) if f.is_file()]
    write(bundle / 'package-manifest.json', dict(model_lock_sha256=LOCK_SHA, files=files))
    archive = out / 'tashu-h13-runtime-v1.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for file in sorted(bundle.rglob('*')):
            if file.is_file():
                z.write(file, file.relative_to(out).as_posix())
    report['archive'] = dict(name=archive.name, bytes=archive.stat().st_size, sha256=digest(archive))
    (out / 'tashu-h13-runtime-v1.zip.sha256').write_text(report['archive']['sha256'] + '  ' + archive.name + '\n', encoding='ascii')
    write(out / 'model-publication.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
