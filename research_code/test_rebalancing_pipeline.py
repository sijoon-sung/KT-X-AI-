import unittest,json,copy
from pathlib import Path
import numpy as np
import rebalancing_pipeline as p

ROOT=Path(__file__).resolve().parents[1]
class Acceptance(unittest.TestCase):
    outputs={}
    @classmethod
    def setUpClass(cls):cls.fixtures=json.loads((ROOT/'research/datasets/H-0015-synthetic-fixtures.json').read_text(encoding='utf-8'))
    def test_main_exact_quantities_and_route(self):
        result=p.plan(copy.deepcopy(self.fixtures['main']));self.outputs['main']=result;expected=self.fixtures['expected'];actions={r['station_id']:r for r in result['quantity_actions']}
        self.assertEqual({k:v['pickup'] for k,v in actions.items() if v['pickup']},expected['main_pickups']);self.assertEqual({k:v['deliver'] for k,v in actions.items() if v['deliver']},expected['main_deliveries'])
        self.assertEqual(result['route']['sequence'],expected['main_route']);self.assertAlmostEqual(result['route']['return_minutes'],7);self.assertEqual(result['route']['peak_load'],6);self.assertFalse(result['dispatch_allowed']);self.assertEqual(result['objective_after']['mean_weighted_buffer_shortfall'],0)
    def test_joint_dependency_changes_cvar_decision(self):
        one=self.fixtures['dependence_case_one'];two=self.fixtures['dependence_case_two'];np.testing.assert_array_equal(np.sort(one['forecast']['stock_scenarios'],1),np.sort(two['forecast']['stock_scenarios'],1))
        a=p.plan(copy.deepcopy(one));b=p.plan(copy.deepcopy(two));self.outputs['joint_case_one']=a;self.outputs['joint_case_two']=b
        destination=lambda r:[x['station_id'] for x in r['quantity_actions'] if x['deliver']]
        self.assertEqual(destination(a),['A']);self.assertEqual(destination(b),['C'])
    def test_expected_only_decision_unchanged_under_same_margins(self):
        for key in ('dependence_case_one','dependence_case_two'):
            request=copy.deepcopy(self.fixtures[key]);request['policy']['risk_weight']=0;result=p.plan(request);self.outputs['expected_only_'+key]=result
            self.assertEqual([x['station_id'] for x in result['quantity_actions'] if x['deliver']],['A'])
    def test_capacity_fallback_preserves_feasibility(self):
        request=copy.deepcopy(self.fixtures['main']);request['fleet']['capacity']=1;result=p.plan(request);self.outputs['capacity_fallback']=result
        self.assertLessEqual(result['route']['peak_load'],1);self.assertLess(result['total_moved_bikes'],10);self.assertTrue(any(not a['route_feasible'] for a in result['attempts']))
    def test_deadline_prevents_unsafe_route(self):
        request=copy.deepcopy(self.fixtures['main']);request['fleet']['max_route_minutes']=1;result=p.plan(request);self.outputs['time_fallback']=result
        self.assertEqual(result['status'],'no_action');self.assertEqual(result['total_moved_bikes'],0);self.assertLessEqual(result['route']['return_minutes'],1)
    def test_unknown_capacity_is_rejected(self):
        request=copy.deepcopy(self.fixtures['main']);request['fleet']['capacity']=None
        with self.assertRaises(ValueError):p.plan(request)
    def test_unsupported_prediction_horizon_is_rejected(self):
        request=copy.deepcopy(self.fixtures['main']);request['forecast']['horizon_minutes']=30
        with self.assertRaises(ValueError):p.plan(request)
    def test_nonfinite_forecast_is_rejected(self):
        request=copy.deepcopy(self.fixtures['main']);request['forecast']['stock_scenarios'][0][0]=float('nan')
        with self.assertRaises(ValueError):p.plan(request)
    def test_unreachable_road_returns_no_action(self):
        request=copy.deepcopy(self.fixtures['main']);request['road_matrix']['duration_minutes'][0][1:]=[None]*4;result=p.plan(request);self.outputs['road_fallback']=result;self.assertEqual(result['status'],'no_action')
    def test_unknown_api_forecast_lineage_is_rejected(self):
        request=copy.deepcopy(self.fixtures['main']);request['mode']='live_shadow';request['forecast']['source_kind']='api_forecast'
        with self.assertRaises(ValueError):p.plan(request)
    def metadata_request(self):
        request=copy.deepcopy(self.fixtures['main']);request['mode']='live_shadow';request['forecast'].update(source_kind='api_forecast',model_lock_sha256='a'*64,model_status='frozen_research_reference',observed_at_utc='2026-10-31T15:00:00+00:00');return request
    def test_stale_collector_metadata_is_rejected(self):
        request=self.metadata_request();request['planner_reference_at_utc']='2026-10-31T15:05:00+00:00'
        with self.assertRaises(ValueError):p.plan(request)
    def test_future_dated_collector_metadata_is_rejected(self):
        request=self.metadata_request();request['planner_reference_at_utc']='2026-10-31T14:59:00+00:00'
        with self.assertRaises(ValueError):p.plan(request)
    def test_collector_clock_metadata_does_not_use_remote_wall_clock(self):
        # Synthetic adapter metadata only, not a real API experiment or hash attestation.
        request=self.metadata_request();request['planner_reference_at_utc']='2026-10-31T15:00:30+00:00';result=p.plan(request);self.assertEqual(result['total_moved_bikes'],10);self.assertFalse(result['dispatch_allowed'])

if __name__=='__main__':unittest.main()
