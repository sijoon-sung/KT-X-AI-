"""Recover summary serialization from immutable H19 checkpoints on dislab.

No rerun, changed observations, routes, plan or decision rule. Preserve the
original numpy integer JSON failure and early out-of-period auxiliary run.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import platform
import numpy as np
from live_api_service import read,atomic


def finalize(folder):
    folder=Path(folder);engineering=read(folder/'engineering.json');states=read(folder/'shadow-progress.json')
    if len(states)!=3:raise ValueError('Require all three preregistered observations, without rerunning them.')
    if [r.get('recorded_at_utc') for r in states]!=['2026-11-01T06:50:00+00:00','2026-11-01T07:00:00+00:00','2026-11-01T07:10:00+00:00']:
        raise ValueError('Unexpected primary observation periods; retain originals for review.')
    failures=int(sum(r['violations'] for r in states));strict=int(sum(bool('net_value'in r and r['legacy_net_value'] is not None and r['net_value']>r['legacy_net_value']+1e-8) for r in states))
    result=dict(started_at_utc=datetime.fromtimestamp((folder/'engineering.json').stat().st_ctime,timezone.utc).isoformat(),
        start_time_source='Engineering checkpoint filesystem creation time, approximate run start',finished_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(),numpy=np.__version__,device='dislab CPU'),engineering=engineering,states=states,
        acceptance_failures=failures,strict_legacy_improvements=strict,primary_summary_serialization_failures=1,
        decision='supported' if failures==0 and strict>0 and engineering['violations']==0 else 'not_supported',
        execution_notes=['Original numpy.int32 strict-count summary could not serialize; this finalizer reads the already persisted engineering/three-state results and changes no routes or criteria.',
            'Earlier auxiliary run opened cached06:40 state before the planned06:50 window; excluded by frozen period, preserved separately, not primary evidence.',
            'Native remote OS clock behind139.523809sec; calibrated estimates and original times are preserved in public metadata.'],
        limitations=['Marginal-model coverage proxy, not observed dispatch effect or lost demand','H19 beam heuristic can miss smaller-capacity incumbents; H20 separates this correction','No API learning,model selection,calibration or new inventory accuracy result'])
    atomic(folder/'summary.json',result);print(dict(states=3,acceptance_failures=failures,strict_legacy_improvements=strict,decision=result['decision']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();finalize(a.folder)
