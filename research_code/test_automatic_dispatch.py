"""Material serving boundaries: source identity, clock, expiry, operator inputs. Run on dislab."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import automatic_dispatch as a


class ServingBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name);self.now=1793515805.
        self.state=dict(source_version='synthetic-v1',recorded_at_utc='2026-11-01T06:50:00+00:00',target_at_utc='2026-11-01T07:00:00+00:00',
            model_label='Flow Matching + GCN',default_ids=['a'],default_depot_id='a',stations=[dict(id='a',name='Synthetic',lat=36.35,lon=127.38,current=6,samples=[6]*128,forecast=dict(median=6,low=6,high=6,p0=0))])
        a.atomic(self.folder/'state.json',self.state);a.atomic(self.folder/'status.json',dict(latest_state='state.json'))
        self.body=dict(source_version='synthetic-v1',automatic=True)
    def tearDown(self):self.tmp.cleanup()
    def request(self,body=None):return a.rpc(self.folder,dict(body=self.body if body is None else body))
    def test_wrong_version_does_not_call_solver(self):
        with patch.object(a,'reference_now',return_value=self.now),patch.object(a,'optimize') as solver:
            result=self.request(dict(source_version='old'));self.assertEqual(result['status'],409);solver.assert_not_called()
            self.assertEqual(result['body']['current_info']['source_version'],'synthetic-v1')
            self.assertNotIn('samples',result['body']['current_info']['stations'][0])
    def test_authoritative_display_does_not_call_solver(self):
        with patch.object(a,'reference_now',return_value=self.now),patch.object(a,'optimize') as solver:
            result=a.rpc(self.folder,dict(path='/api/dispatch/info'));self.assertEqual(result['status'],200);solver.assert_not_called()
            self.assertEqual(result['body']['source_version'],'synthetic-v1')
    def test_expired_does_not_call_solver(self):
        with patch.object(a,'reference_now',return_value=self.now+600),patch.object(a,'optimize') as solver:
            self.assertEqual(self.request()['status'],409);solver.assert_not_called()
    def test_future_observation_rejects_clock(self):
        with patch.object(a,'reference_now',return_value=self.now-60),patch.object(a,'optimize') as solver:
            self.assertEqual(self.request()['status'],409);solver.assert_not_called()
    def test_old_observation_rejects(self):
        with patch.object(a,'reference_now',return_value=self.now+800),patch.object(a,'optimize') as solver:
            self.assertEqual(self.request()['status'],409);solver.assert_not_called()
    def test_calculation_latency_cannot_extend_horizon(self):
        result=dict(route=dict(return_minutes=5))
        with patch.object(a,'reference_now',side_effect=[self.now,self.now+600]),patch.object(a,'optimize',return_value=result):
            self.assertEqual(self.request()['status'],409)
    def test_valid_issue_records_calibrated_time(self):
        result=dict(route=dict(return_minutes=5))
        with patch.object(a,'reference_now',side_effect=[self.now,self.now+1,self.now+1]),patch.object(a,'optimize',return_value=result):
            response=self.request();self.assertEqual(response['status'],200)
            self.assertEqual(response['body']['issued_at_utc'],datetime.fromtimestamp(self.now+1,timezone.utc).isoformat())
    def test_unsafe_operator_inputs_reject(self):
        for body in [dict(capacity=True),dict(capacity=0),dict(capacity=13),dict(budget=-1),dict(target=0),dict(deadline_minutes=60),
                     dict(station_ids=['a','a']),dict(station_ids=['missing']),dict(depot_id='missing'),dict(automatic='yes'),dict(objective='free'),dict(token='never')]:
            with self.subTest(body=body),self.assertRaises(ValueError):a.parameters(self.state,body)


if __name__=='__main__':unittest.main()
