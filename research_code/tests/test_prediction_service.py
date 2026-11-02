"""Synthetic service contracts; execute on dislab, no real API used here."""
import json,sys,threading,unittest,urllib.request,urllib.error
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prediction_service as s
import history_to_stock as h


def engine():
    rates=np.zeros((7,24,2,2));rates[:]=[2.,1.]
    return s.ForecastEngine(['A','B'],{'gcn':rates,'calendar':rates.copy()},h.numeric_contract([3])['cutoff'],{'A':'예시 대여소'})


class PredictionServiceTests(unittest.TestCase):
    def test_prediction_domain_horizon_and_repeatability(self):
        e=engine();a=e.predict('A',1,'2026-09-30T07:00:00+09:00');b=e.predict('A',1,'2026-09-30T07:00:00+09:00')
        self.assertEqual(a['forecast_id'],b['forecast_id']);self.assertEqual(a['models'],b['models'])
        self.assertEqual(a['target_at'],'2026-09-30T07:30:00+09:00')
        for value in a['models'].values():
            self.assertGreaterEqual(value['point_prediction'],0)
            self.assertTrue(0<=value['zero_probability']<=1)
            self.assertAlmostEqual(sum(x['probability'] for x in value['distribution']),1,places=10)
            self.assertLessEqual(value['interval']['lower'],value['point_prediction'])
            self.assertGreaterEqual(value['interval']['upper'],value['point_prediction'])

    def test_timezone_and_non_clock_scenario(self):
        e=engine();a=e.predict('A',2,'2026-09-29T22:00:00+00:00');b=e.predict('A',2,'2026-09-30T07:00:00+09:00')
        self.assertEqual(a['models'],b['models'])
        self.assertFalse(e.predict('A',2,'2026-09-30T07:15:00+09:00')['scope']['validated_clock_origin'])

    def test_zero_is_observed_but_missing_is_not_zero(self):
        e=engine();zero=e.predict('A',0,'2026-09-30T07:00:00+09:00')
        self.assertGreater(zero['models']['gcn']['zero_probability'],0)
        for value in (None,-1,.5,True,float('nan')):
            with self.assertRaises(ValueError):e.predict('A',value,'2026-09-30T07:00:00+09:00')

    def test_unknown_station_time_and_observation_contract(self):
        e=engine()
        for station,time in [('NEW','2026-09-30T07:00:00+09:00'),('A','2026-09-30T22:00:00+09:00'),('A','2026-09-30T07:00:00')]:
            with self.assertRaises(ValueError):e.predict(station,1,time)
        with self.assertRaises(ValueError):e.predict('A',1,'2026-09-30T07:00:00+09:00',observed_at='2026-09-30T07:03:00+09:00')

    def test_large_stock_is_not_capped(self):
        result=engine().predict('A',1000,'2026-09-30T07:00:00+09:00')['models']['gcn']
        self.assertGreater(result['point_prediction'],900)
        self.assertAlmostEqual(result['expected_stock'],999.5,places=8)

    def test_snapshot_missing_unknown_duplicate_and_future_fields(self):
        e=engine();value={'observed_at':'2026-09-30T07:00:00+09:00','stations':[{'station_id':'A','parking_count':None},{'station_id':'B','parking_count':0},{'station_id':'NEW','parking_count':3}]}
        result=e.snapshot(value)['forecasts'];self.assertEqual([x['status'] for x in result],['unavailable','ok','unsupported'])
        value['stations'].append({'station_id':'A','parking_count':1})
        with self.assertRaises(ValueError):e.snapshot(value)
        value={'observed_at':'2026-09-30T07:00:00+09:00','stations':[{'station_id':'A','parking_count':1,'future_stock':0}]}
        with self.assertRaises(ValueError):e.snapshot(value)

    def test_future_truth_changes_do_not_change_prediction(self):
        e=engine();replay=s.SeptemberReplay.__new__(s.SeptemberReplay)
        t=int(datetime(2026,9,30,7,tzinfo=h.KST).timestamp())
        replay.times=np.arange(t,t+2400,600);replay.origins=np.array([t]);replay.time_index={int(x):i for i,x in enumerate(replay.times)}
        replay.ids=np.array(['A']);replay.columns={'A':0};replay.values=np.ones((4,1));replay.actual=replay.times[:,None]+1
        a=e.predict(**replay.current(t,'A'));replay.values[3,0]=100
        b=e.predict(**replay.current(t,'A'));self.assertEqual(a['models'],b['models'])
        self.assertEqual(replay.truth(t,'A')['target_parking_count'],100)

    def test_http_contract_and_private_paths(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),s.handler(engine(),None));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(base+'/api/health') as response:self.assertEqual(json.loads(response.read())['status'],'ok')
            value={'station_id':'A','parking_count':1,'issued_at':'2026-09-30T07:00:00+09:00'}
            request=urllib.request.Request(base+'/api/forecast',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:self.assertIn('gcn',json.loads(response.read())['models'])
            value['future_stock']=0
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(urllib.request.Request(base+'/api/forecast',data=json.dumps(value).encode()))
            self.assertEqual(error.exception.code,400)
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+'/research/hypotheses/H-0008.json')
            self.assertEqual(error.exception.code,404)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
