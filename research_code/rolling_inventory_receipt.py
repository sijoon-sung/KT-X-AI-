"""Collector-clock confirmation after remote inference, before future truth.

Preserve the original artifact. Only provenance/timestamp fields change; no
model is executed here. The collector clock also timestamps API observations.
"""
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from rolling_inventory_metrics import LABELS
import historical_calendar as h

def confirm(source,out,confirmed_at=None):
    source=Path(source);out=Path(out)
    if out.exists() or source.resolve()==out.resolve():raise ValueError('Preserve original and prior confirmation artifacts.')
    with np.load(source,allow_pickle=False) as f:values={k:f[k].copy() for k in f.files}
    confirmed=time.time() if confirmed_at is None else float(confirmed_at)
    if not np.isfinite(confirmed) or confirmed<float(np.max(values['actual'])) or confirmed>=int(values['target_scheduled']):raise ValueError('Collector confirmation must follow current observation and precede target.')
    if int(values['target_scheduled'])-int(values['scheduled'])!=600:raise ValueError('Unexpected forecast horizon.')
    original_sha=h.digest(source);sample_sha=hashlib.sha256(b''.join(values[k].tobytes() for k in LABELS)).hexdigest()
    meta=json.loads(str(values['metadata_json']));meta.update(remote_os_issued_at_unix_seconds=float(values['issued_at_unix_seconds']),collector_confirmed_at_unix_seconds=confirmed,
        issue_clock_basis='collector confirmation after remote inference receipt; same clock as API observations',original_forecast_sha256=original_sha,issued_at_utc=datetime.fromtimestamp(confirmed,timezone.utc).isoformat())
    values['issued_at_unix_seconds']=np.float64(confirmed);values['metadata_json']=np.array(json.dumps(meta,ensure_ascii=False))
    np.savez_compressed(out,**values)
    return dict(original_forecast_sha256=original_sha,confirmed_forecast_sha256=h.digest(out),sample_values_sha256=sample_sha,collector_confirmed_at_unix_seconds=confirmed,
        target_scheduled_at_unix_seconds=int(values['target_scheduled']),prediction_values_changed=False,future_truth_argument=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forecast',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(confirm(a.forecast,a.out)))
