"""H18 semantic acceptance; run on dislab, no credential or HTTP output."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from live_api_service import Clock,atomic,groups,reference_now,rpc,NoRedirect
from rolling_inventory_metrics import LABELS


class LiveSemantics(unittest.TestCase):
    def test_clock_uncertainty_is_bounded(self):
        with self.assertRaises(ValueError):Clock(137,2.1)
        clock=Clock(137,.2);self.assertAlmostEqual(clock.now()-time.time(),137,delta=.1)

    def test_monotonic_clock_ignores_native_wall_jump(self):
        clock=Clock(137,.2);before=clock.now()
        with patch('live_api_service.time.time',return_value=0):self.assertGreaterEqual(clock.now(),before)

    def test_zero_recovery_and_new_empty_groups(self):
        y=np.array([0,0,1]);current=np.array([0,2,0]);draws={k:np.repeat(current[:,None],128,axis=1) for k in LABELS}
        result=groups(draws,y,current,np.ones(3,bool))
        self.assertEqual(result['zero_to_positive'],1);self.assertEqual(result['positive_to_zero'],1)
        self.assertEqual(result['groups']['current_zero']['recent_cfm']['pairs'],2)

    def test_atomic_missing_is_not_a_zero_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.json';atomic(p,dict(value=None));self.assertIsNone(json.loads(p.read_text(encoding='utf-8'))['value'])
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_redirect_does_not_forward_auth(self):
        self.assertIsNone(NoRedirect().redirect_request(None,None,None,None,None,None))

    def test_expired_clock_anchor_rejected(self):
        with self.assertRaises(ValueError):reference_now(dict(clock=dict(utc_anchor=0,monotonic_anchor=time.monotonic()-90000)))

    def test_source_version_mismatch_prevents_planning(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);clock=Clock(0,.1);atomic(folder/'status.json',dict(clock=clock.metadata(),latest_state='state.json'))
            atomic(folder/'state.json',dict(source_version='new',recorded_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
            with patch('dispatch_engine.optimize') as calc:
                result=rpc(folder,dict(path='/api/dispatch/optimize',body=dict(source_version='old')))
                self.assertEqual(result['status'],409);calc.assert_not_called()

    def test_stale_observation_prevents_planning(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);clock=Clock(0,.1);atomic(folder/'status.json',dict(clock=clock.metadata(),latest_state='state.json'))
            atomic(folder/'state.json',dict(source_version='new',recorded_at_utc='2020-01-01T00:00:00+00:00'))
            with patch('dispatch_engine.optimize') as calc:
                result=rpc(folder,dict(path='/api/dispatch/optimize',body=dict(source_version='new')))
                self.assertEqual(result['status'],409);calc.assert_not_called()


if __name__=='__main__':unittest.main()
