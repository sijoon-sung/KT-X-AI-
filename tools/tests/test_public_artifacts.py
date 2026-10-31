"""Verify public bytes against preregistered provenance after a real Git checkout."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublishedArtifactTests(unittest.TestCase):
    def test_nested_source_catalog_matches_frozen_manifest_bytes(self):
        for path in (ROOT / 'research/datasets').glob('*.json'):
            manifest = load(path)
            if manifest.get('source_catalog'):
                catalog = (ROOT / manifest['source_catalog']).resolve()
                self.assertTrue(catalog.is_relative_to(ROOT.resolve()))
                self.assertEqual(digest(catalog), manifest['source_catalog_sha256'])

    def test_frozen_input_manifests_match_actual_public_bytes(self):
        available = {digest(path) for path in (ROOT / 'research/datasets').glob('*.json')}
        for path in (ROOT / 'research/hypotheses').glob('H-*.json'):
            record = load(path)
            if record.get('preregistration', {}).get('plan_sha256'):
                with self.subTest(hypothesis=record['id']):
                    self.assertIn(record['plan']['data']['input_manifest_sha256'], available)

    def test_public_result_artifacts_match_recorded_bytes(self):
        for path in (ROOT / 'research/hypotheses').glob('H-*.json'):
            for name in (load(path).get('result') or {}).get('run_manifests', []):
                manifest_path = ROOT / name
                for artifact in load(manifest_path).get('artifacts', []):
                    if artifact.get('availability', '').startswith('public') and artifact.get('sha256'):
                        public_path = (manifest_path.parent / artifact['id']).resolve()
                        with self.subTest(artifact=artifact['id'], run=name):
                            self.assertTrue(public_path.is_relative_to(ROOT.resolve()))
                            self.assertEqual(digest(public_path), artifact['sha256'])


if __name__ == '__main__':
    unittest.main()
