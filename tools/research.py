"""Research records and PR checks. Python standard library only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r'H-\d{4}')
HASH = re.compile(r'[0-9a-f]{64}')
STATUSES = {'planned', 'preregistered', 'running', 'completed', 'withdrawn'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def plan_hash(plan):
    value = json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def moment(value):
    if not isinstance(value, str):
        raise ValueError('Timestamp must be an ISO string.')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def record_path(identifier, root=ROOT):
    if not ID.fullmatch(identifier) or identifier == 'H-0000':
        raise ValueError('Use H-0001..H-9999; H-0000 is reserved for templates.')
    return root / 'research' / 'hypotheses' / (identifier + '.json')


def validate(record, path, root=ROOT):
    if not isinstance(record, dict):
        return ['Research record must be a JSON object.']
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    identifier = record.get('id', '')
    require(isinstance(identifier, str) and bool(ID.fullmatch(identifier)) and path.stem == identifier, 'Record ID must match its filename.')
    require(record.get('schema_version') == 1, 'schema_version must be 1.')
    require(bool(record.get('title')), 'Title is required.')
    status = record.get('status')
    require(isinstance(status, str) and status in STATUSES, 'Unknown status.')
    plan = record.get('plan', {})
    if not isinstance(plan, dict) or not isinstance(record.get('preregistration', {}), dict):
        return errors + ['Plan and preregistration must be objects.']
    strict = status not in ('planned', 'withdrawn') or bool(record.get('preregistration', {}).get('plan_sha256'))
    try:
        for key in ('question', 'hypothesis'):
            require(bool(plan[key]), 'Missing plan.' + key)
        require(bool(plan['prior_art']), 'At least one prior-art record is required.')
        if not isinstance(plan['prior_art'], list) or not all(isinstance(x, dict) for x in plan['prior_art']):
            raise ValueError('prior_art must be a list of objects.')
        for ref in plan['prior_art']:
            require(all(ref.get(k) for k in ('title', 'url', 'verified_scope', 'difference_to_test')), 'Prior art needs source, verified scope and comparison.')
        data = plan['data']
        require(all(data.get(k) for k in ('dataset_id', 'observed_truth', 'not_observed', 'input_manifest_sha256')), 'Data provenance and truth scope are required.')
        ev = plan['evaluation']
        for key in ('design', 'periods', 'baselines', 'primary_metric', 'success_rule', 'failure_rule', 'exclusion_rule', 'uncertainty_plan', 'seed_plan'):
            require(bool(ev.get(key)), 'Missing evaluation.' + key)
        require(ev['primary_metric'].get('direction') in ('lower', 'higher'), 'Metric direction must be lower or higher.')
        require(all(ev['primary_metric'].get(k) for k in ('name', 'unit')), 'Primary metric needs name and unit.')
        if strict:
            require('TODO' not in json.dumps(plan, ensure_ascii=False), 'Fill every TODO before freezing.')
            require(bool(HASH.fullmatch(data['input_manifest_sha256'])), 'Input manifest requires a SHA-256.')
            require(all(ref['url'].startswith('https://') for ref in plan['prior_art']), 'Use https primary-source links.')
            periods = {}
            for period in ev['periods']:
                start, end = moment(period['start']), moment(period['end'])
                require(start < end, 'Period start must precede end.')
                require(period['role'] not in periods, 'Period roles must be unique.')
                periods[period['role']] = (start, end)
            ordered = [periods[k] for k in ('train', 'validation', 'test') if k in periods]
            require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), 'Train/validation/test periods overlap.')
    except (KeyError, TypeError, ValueError) as exc:
        errors.append('Invalid plan: ' + str(exc))
    frozen = record.get('preregistration', {})
    if status not in ('planned', 'withdrawn') or frozen.get('plan_sha256'):
        require(frozen.get('plan_sha256') == plan_hash(plan), 'Frozen plan changed or its hash is missing.')
        try:
            stamp = moment(frozen['frozen_at_utc'])
            require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0, 'Freeze timestamp must be UTC.')
        except (KeyError, TypeError, ValueError):
            errors.append('Freeze timestamp is missing/invalid.')
    if status == 'withdrawn':
        require(bool(record.get('withdrawal_reason')), 'Withdrawal needs a reason.')
    if status == 'completed':
        result = record.get('result') or {}
        require(result.get('decision') in ('supported', 'not_supported', 'inconclusive'), 'Completion requires an evidence-based decision.')
        require(isinstance(result.get('adopt_code'), bool), 'Completion requires adopt_code true/false.')
        require(bool(result.get('summary')) and isinstance(result.get('limitations'), list), 'Completion needs summary and limitations.')
        manifests = result.get('run_manifests', [])
        require(bool(manifests), 'Completion needs at least one run manifest.')
        for name in manifests:
            try:
                runpath = (root / name).resolve()
                require(runpath.is_relative_to((root / 'research' / 'runs').resolve()), 'Run manifests must stay inside research/runs.')
                if not runpath.is_relative_to((root / 'research' / 'runs').resolve()):
                    continue
                run = load(runpath)
                require(run.get('schema_version') == 1, 'Run schema_version must be 1.')
                require(run.get('hypothesis_id') == identifier, 'Run points to another hypothesis.')
                require(run.get('plan_sha256') == frozen.get('plan_sha256'), 'Run plan hash does not match.')
                require(run.get('input_manifest_sha256') == plan['data']['input_manifest_sha256'], 'Run input differs from preregistered input manifest.')
                require(bool(re.fullmatch(r'[0-9a-f]{40}', run.get('code_commit', ''))), 'Run needs a full code commit.')
                require('TODO' not in json.dumps(run), 'Run still contains TODO.')
                start, end = moment(run['started_at_utc']), moment(run['finished_at_utc'])
                require(start.utcoffset() is not None and end.utcoffset() is not None, 'Run timestamps need timezones.')
                require(moment(frozen['frozen_at_utc']) <= start <= end, 'Run must follow the frozen plan.')
                require(isinstance(run.get('command'), list) and bool(run['command']), 'Run command must be an argument list.')
                require(bool(run.get('environment')) and bool(run.get('evaluation_scope')), 'Run environment and evaluation scope are required.')
                metrics = run.get('metrics', {})
                require(plan['evaluation']['primary_metric']['name'] in metrics, 'Run lacks the primary metric.')
                require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in metrics.values()), 'Run metrics must be finite numbers.')
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append('Invalid run manifest: ' + str(exc))
    return errors


def frozen_change_errors(old, new):
    if old and old.get('preregistration', {}).get('plan_sha256'):
        if old['plan'] != new['plan'] or old['preregistration'] != new['preregistration']:
            return ['A previously frozen plan or registration was rewritten; create another hypothesis ID.']
    return []


def history_errors(root, name):
    """Check every reachable file revision, including commits inside one open PR."""
    if not (root / '.git').exists():
        return []
    commits = subprocess.check_output(['git', 'rev-list', '--reverse', '--topo-order', 'HEAD', '--', name], cwd=root, text=True).splitlines()
    frozen, errors = None, []
    for commit in commits:
        blob = subprocess.run(['git', 'show', commit + ':' + name], cwd=root, text=True, capture_output=True, encoding='utf-8')
        if blob.returncode:
            if frozen:
                errors.append(name + ': frozen record was deleted in ' + commit[:8])
            continue
        value = json.loads(blob.stdout)
        if frozen:
            errors.extend(name + '@' + commit[:8] + ': ' + e for e in frozen_change_errors(frozen, value))
        elif value.get('preregistration', {}).get('plan_sha256'):
            frozen = value
    current = root / name
    if frozen and current.exists():
        errors.extend(name + ': ' + e for e in frozen_change_errors(frozen, load(current)))
    return errors


def pr_errors(title, changed, records, draft):
    match = re.match(r'^\[(H-\d{4}|workflow|maintenance)\]\s+\S', title)
    if not match:
        return ['PR title must begin [H-0001], [workflow], or [maintenance].']
    kind = match[1]
    hypotheses = [p for p in changed if re.fullmatch(r'research/hypotheses/H-\d{4}\.json', p)]
    if kind.startswith('H-'):
        expected = 'research/hypotheses/' + kind + '.json'
        errors = [] if hypotheses == [expected] else ['A hypothesis PR must change exactly its own hypothesis record.']
        record = records.get(kind)
        if not record:
            errors.append('Hypothesis record cannot be deleted/missing.')
        elif not draft and record['status'] not in ('completed', 'withdrawn'):
            errors.append('Keep this hypothesis PR in Draft until completed or withdrawn.')
        return errors
    if hypotheses:
        return ['Research records belong in hypothesis PRs.']
    workflow_allowed = lambda p: p in ('AGENTS.md', 'README.md', '.gitignore', 'research/README.md') or p.startswith(('.github/', 'docs/', 'research/templates/', 'tools/research.py', 'tools/tests/'))
    if kind == 'workflow' and any(not workflow_allowed(p) for p in changed):
        return ['Workflow PR contains non-workflow files.']
    if kind == 'maintenance' and any(p.startswith(('src/', 'tashu/', 'scripts/', 'tests/')) and p.endswith('.py') for p in changed):
        return ['Model/data behavior changes need a hypothesis PR.']
    return []


def check(root=ROOT, base_ref=None, title=None, draft=True):
    records, errors = {}, []
    for path in sorted((root / 'research' / 'hypotheses').glob('H-*.json')):
        try:
            record = load(path)
            records[record['id']] = record
            errors.extend(path.name + ': ' + e for e in validate(record, path, root))
            errors.extend(history_errors(root, path.relative_to(root).as_posix()))
        except (OSError, KeyError, ValueError) as exc:
            errors.append(path.name + ': ' + str(exc))
    if base_ref:
        base = subprocess.check_output(['git', 'rev-parse', '--verify', base_ref], cwd=root, text=True).strip()
        changed = subprocess.check_output(['git', 'diff', '--name-only', base + '...HEAD'], cwd=root, text=True).splitlines()
        for name in changed:
            if not re.fullmatch(r'research/hypotheses/H-\d{4}\.json', name):
                continue
            path = root / name
            if not path.exists():
                errors.append('Hypothesis records cannot be deleted: ' + name)
                continue
            old = subprocess.run(['git', 'show', base + ':' + name], cwd=root, text=True, capture_output=True)
            if old.returncode == 0:
                errors.extend(frozen_change_errors(json.loads(old.stdout), load(path)))
        if title:
            errors.extend(pr_errors(title, changed, records, draft))
    elif title:
        errors.append('--pr-title requires --base-ref.')
    return errors, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    new = sub.add_parser('new'); new.add_argument('id'); new.add_argument('--title', required=True)
    freeze = sub.add_parser('freeze'); freeze.add_argument('id')
    check_parser = sub.add_parser('check'); check_parser.add_argument('--base-ref'); check_parser.add_argument('--pr-title'); check_parser.add_argument('--draft', choices=['true', 'false'], default='true')
    sub.add_parser('list')
    args = ap.parse_args()
    try:
        if args.command == 'new':
            path = record_path(args.id)
            if path.exists(): raise ValueError('Hypothesis already exists.')
            record = load(ROOT / 'research' / 'templates' / 'hypothesis.json')
            record.update(id=args.id, title=args.title); save(path, record)
            print('Created ' + str(path.relative_to(ROOT)) + '; fill the plan, open a Draft PR, then freeze before any experiment.')
            return
        if args.command == 'freeze':
            path = record_path(args.id); record = load(path)
            if record['status'] != 'planned': raise ValueError('Only planned records can be frozen once.')
            record['status'] = 'preregistered'
            record['preregistration'] = {'plan_sha256': plan_hash(record['plan']), 'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
            errors = validate(record, path)
            if errors: raise ValueError('\n'.join(errors))
            save(path, record)
            print('Frozen ' + args.id + '; commit and push this plan to its Draft PR before experiments.')
            return
        errors, records = check(base_ref=getattr(args, 'base_ref', None), title=getattr(args, 'pr_title', None), draft=getattr(args, 'draft', 'true') == 'true')
        for error in errors: print('ERROR: ' + error, file=sys.stderr)
        if errors: raise SystemExit(1)
        for identifier, record in records.items():
            print(identifier, record['status'], record['title'], (record.get('result') or {}).get('decision', 'unexecuted'))
        print('Validated ' + str(len(records)) + ' research records. This is a protocol check, not empirical validation.')
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        ap.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
