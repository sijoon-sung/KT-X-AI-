import json,sys,threading,unittest,urllib.request,urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import advanced_prediction_service as a
import prediction_service as b
import scaled_flow_core as c
import history_to_stock as h


def engine():
    ids=np.array(['A']);rates=np.ones((7,24,1,2))
    reference=b.ForecastEngine(ids,dict(gcn=rates,calendar=rates),h.numeric_contract([2])['cutoff'])
    delta=np.tile(np.array([-1,0,1,2]),128).reshape(1,1,512);depth=np.maximum(-delta,0)
    expanded=np.ones((c.SLOTS,1,2))
    return a.AdvancedEngine(reference,ids,{key:expanded for key in c.KINDS},reference.cutoff,[int(c.context_slot(9,2,7,0))],delta,depth,{})


class AdvancedServiceTests(unittest.TestCase):
    def test_fixed_fusion_ensemble_and_reference_are_separate(self):
        e=engine();r=e.predict('A',2,'2026-09-30T07:00:00+09:00')
        self.assertEqual(r['primary_model'],'large_seoul_cfm');p=r['models']['large_seoul_cfm']
        self.assertEqual(p['point_prediction'],2);self.assertEqual(p['expected_stock'],2.5)
        self.assertAlmostEqual(sum(x['probability'] for x in p['distribution']),1)
        self.assertIn('gcn',r['models']);self.assertIn('calendar',r['models'])

    def test_zero_current_reflects_without_missing_imputation(self):
        e=engine();r=e.predict('A',0,'2026-09-30T07:00:00+09:00')['models']['large_seoul_cfm']
        self.assertEqual(r['zero_probability'],.5);self.assertEqual(r['point_prediction'],0)
        with self.assertRaises(ValueError):e.predict('A',None,'2026-09-30T07:00:00+09:00')
        with self.assertRaises(ValueError):e.predict('A',1,'2026-09-30T07:00:00+09:00',model_variant='unregistered')

    def test_manual_model_choice_repeatability_and_high_stock(self):
        e=engine();first=e.predict('A',1000,'2026-09-30T07:00:00+09:00');second=e.predict('A',1000,'2026-09-30T07:00:00+09:00')
        self.assertEqual(first['forecast_id'],second['forecast_id']);self.assertEqual(first['models'],second['models'])
        self.assertGreater(first['models']['large_seoul_cfm']['point_prediction'],900)
        self.assertEqual(e.predict('A',1,'2026-09-30T07:00:00+09:00',model_variant='large')['primary_model'],'large')

    def test_http_keeps_future_labels_out_and_base_available(self):
        e=engine();server=ThreadingHTTPServer(('127.0.0.1',0),a.handler(e.reference,None,e));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:'+str(server.server_port)
        try:
            value=dict(station_id='A',parking_count=1,issued_at='2026-09-30T07:00:00+09:00')
            for path,model in [('/api/forecast','gcn'),('/api/advanced/forecast','large_seoul_cfm')]:
                request=urllib.request.Request(url+path,data=json.dumps(value).encode())
                with urllib.request.urlopen(request) as response:self.assertEqual(json.loads(response.read())['primary_model'],model)
            value['target_stock']=3
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(urllib.request.Request(url+'/api/advanced/forecast',data=json.dumps(value).encode()))
            self.assertEqual(error.exception.code,400)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
