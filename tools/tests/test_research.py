import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('research', ROOT / 'tools' / 'research.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def record():
    data = r.load(ROOT / 'research' / 'templates' / 'hypothesis.json')
    data.update(id='H-0001', title='Synthetic protocol test only')
    def fill(value):
        if isinstance(value, dict): return {k: fill(v) for k, v in value.items()}
        if isinstance(value, list): return [fill(v) for v in value]
        return 'specified test detail' if isinstance(value, str) and 'TODO' in value else value
    data['plan'] = fill(data['plan'])
    data['plan']['prior_art'][0]['url'] = 'https://example.org/primary-source'
    data['plan']['data']['input_manifest_sha256'] = 'a' * 64
    data['plan']['evaluation']['periods'] = [{'role': 'train', 'start': '2026-01-01', 'end': '2026-02-01'}, {'role': 'test', 'start': '2026-02-01', 'end': '2026-03-01'}]
    data['status'] = 'preregistered'
    data['preregistration'] = {'plan_sha256': r.plan_hash(data['plan']), 'frozen_at_utc': '2026-01-01T00:00:00+00:00'}
    return data


class ProtocolTests(unittest.TestCase):
    def test_valid_preregistered_plan(self):
        self.assertEqual(r.validate(record(), Path('H-0001.json')), [])

    def test_unfilled_plan_cannot_freeze(self):
        data = r.load(ROOT / 'research' / 'templates' / 'hypothesis.json')
        data.update(id='H-0001', status='preregistered')
        self.assertTrue(any('TODO' in e for e in r.validate(data, Path('H-0001.json'))))

    def test_mutation_and_reset_of_frozen_plan_are_rejected(self):
        old = record(); new = copy.deepcopy(old)
        new['plan']['hypothesis'] = 'Changed after seeing results'; new['status'] = 'planned'
        self.assertTrue(r.frozen_change_errors(old, new))
        self.assertTrue(any('Frozen plan' in e for e in r.validate(new, Path('H-0001.json'))))

    def test_overlapping_split_is_rejected(self):
        data = record(); data['plan']['evaluation']['periods'][1]['start'] = '2026-01-20'
        data['preregistration']['plan_sha256'] = r.plan_hash(data['plan'])
        self.assertTrue(any('overlap' in e for e in r.validate(data, Path('H-0001.json'))))

    def test_results_need_execution_evidence(self):
        data = record(); data['status'] = 'completed'
        data['result'] = {'decision': 'not_supported', 'adopt_code': False, 'summary': 'No improvement.', 'limitations': [], 'run_manifests': []}
        self.assertTrue(any('run manifest' in e for e in r.validate(data, Path('H-0001.json'))))

    def test_wrong_data_and_early_execution_are_rejected(self):
        data = record(); data['status'] = 'completed'
        name = 'research/runs/test/manifest.json'
        data['result'] = {'decision': 'inconclusive', 'adopt_code': False, 'summary': 'Protocol fixture only.', 'limitations': [], 'run_manifests': [name]}
        manifest = {'schema_version': 1, 'hypothesis_id': data['id'], 'plan_sha256': data['preregistration']['plan_sha256'], 'input_manifest_sha256': 'b'*64,
                    'code_commit': 'c'*40, 'started_at_utc': '2025-12-01T00:00:00+00:00', 'finished_at_utc': '2025-12-02T00:00:00+00:00',
                    'command': ['python', 'experiment.py'], 'environment': {'python': '3.12'}, 'evaluation_scope': 'fixture', 'metrics': {'specified test detail': .2}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); r.save(root/name, manifest)
            errors = r.validate(data, Path('H-0001.json'), root)
            self.assertTrue(any('input differs' in e for e in errors))
            self.assertTrue(any('follow the frozen' in e for e in errors))
            manifest.update(input_manifest_sha256='a'*64, started_at_utc='2026-01-02T00:00:00+00:00', finished_at_utc='2026-01-02T01:00:00+00:00')
            r.save(root/name, manifest)
            self.assertEqual(r.validate(data, Path('H-0001.json'), root), [])

    def test_manifest_cannot_escape_run_directory(self):
        data = record(); data['status'] = 'completed'
        data['result'] = {'decision': 'inconclusive', 'adopt_code': False, 'summary': 'Fixture.', 'limitations': [], 'run_manifests': ['../outside.json']}
        self.assertTrue(any('inside research/runs' in e for e in r.validate(data, Path('H-0001.json'))))

    def test_one_hypothesis_per_pr_and_draft_lifecycle(self):
        records = {'H-0001': record()}; path = 'research/hypotheses/H-0001.json'
        self.assertEqual(r.pr_errors('[H-0001] Study', [path], records, True), [])
        self.assertTrue(r.pr_errors('[H-0001] Study', [path], records, False))
        self.assertTrue(r.pr_errors('[H-0001] Study', [path, 'research/hypotheses/H-0002.json'], records, True))

    def test_workflow_cannot_hide_model_changes(self):
        self.assertEqual(r.pr_errors('[workflow] Templates', ['AGENTS.md', 'tools/research.py'], {}, False), [])
        self.assertTrue(r.pr_errors('[workflow] Templates', ['tashu/demand.py'], {}, False))
        self.assertTrue(r.pr_errors('[maintenance] Rename', ['src/model.py'], {}, False))

    def test_record_id_cannot_be_a_path(self):
        with self.assertRaises(ValueError): r.record_path('../H-0001')

    def test_frozen_plan_changes_within_first_unmerged_pr_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); name = 'research/hypotheses/H-0001.json'
            def git(*args):
                subprocess.run(['git', '-c', 'user.name=Protocol test', '-c', 'user.email=protocol@example.invalid', *args], cwd=root, check=True, capture_output=True)
            git('init')
            data = record(); r.save(root/name, data)
            git('add', name); git('commit', '-m', 'Freeze fixture')
            self.assertEqual(r.history_errors(root, name), [])
            data['plan']['hypothesis'] = 'A different plan'
            data['preregistration']['plan_sha256'] = r.plan_hash(data['plan'])
            r.save(root/name, data); git('add', name); git('commit', '-m', 'Rewrite fixture plan')
            self.assertTrue(r.history_errors(root, name))


if __name__ == '__main__': unittest.main()
