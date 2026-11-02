import json,tempfile,unittest
from pathlib import Path
import numpy as np
from rolling_inventory_metrics import LABELS
from rolling_inventory_receipt import confirm

class Receipt(unittest.TestCase):
    def make(self,path):
        np.savez_compressed(path,actual=np.array([901.]),scheduled=np.int64(900),target_scheduled=np.int64(1500),issued_at_unix_seconds=np.float64(800),metadata_json=np.array('{}'),**{k:np.array([[0,1]]) for k in LABELS})
    def test_preserve_samples_and_original_remote_clock(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original.npz';out=root/'confirmed.npz';self.make(source);report=confirm(source,out,1000)
            self.assertFalse(report['prediction_values_changed'])
            with np.load(source) as a,np.load(out) as b:
                for label in LABELS:np.testing.assert_array_equal(a[label],b[label])
                self.assertEqual(float(a['issued_at_unix_seconds']),800);self.assertEqual(float(b['issued_at_unix_seconds']),1000)
                self.assertEqual(json.loads(str(b['metadata_json']))['remote_os_issued_at_unix_seconds'],800)
    def test_reject_after_target_or_before_current(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original.npz';self.make(source)
            for confirmed in (900,1500,1600):
                with self.assertRaises(ValueError):confirm(source,root/'out.npz',confirmed)

if __name__=='__main__':unittest.main()
