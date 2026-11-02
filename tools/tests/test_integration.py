"""Integration must preserve exact experiment blobs from recorded ancestors."""
import json,subprocess,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research as r

class IntegrationTests(unittest.TestCase):
    def test_source_integrity_and_missing_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run=lambda *args:subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
            run('init');run('config','user.name','Protocol Test');run('config','user.email','test@example.invalid')
            path=root/'research_code/model.py';path.parent.mkdir();path.write_text('original\n')
            run('add','.');run('commit','-m','source');sha=run('rev-parse','HEAD')
            self.assertTrue(r.integration_errors(root,['research_code/model.py']))
            doc=root/'docs';doc.mkdir();(doc/'github-integration.json').write_text(json.dumps({'sources':[{'commit':sha,'pr':'https://github.com/sijoon-sung/KT-X-AI-/pull/31'}]}))
            run('add','.');run('commit','-m','integration')
            self.assertEqual(r.integration_errors(root,['research_code/model.py']),[])
            path.write_text('changed numerical behavior\n');run('add','.');run('commit','-m','tamper')
            self.assertTrue(r.integration_errors(root,['research_code/model.py']))
            path.unlink();run('add','.');run('commit','-m','delete')
            self.assertTrue(r.integration_errors(root,['research_code/model.py']))
