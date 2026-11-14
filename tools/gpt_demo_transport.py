"""Use the signed-in Codex CLI for the local GPT product demo."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


MODEL = 'gpt-6.1-sol'


def generate(request, workspace):
    workspace = Path(workspace).resolve(strict=True)
    with TemporaryDirectory(prefix='gpt-', dir=workspace) as folder:
        schema = Path(folder) / 'answer.json'
        output = Path(folder) / 'result.json'
        schema.write_text(json.dumps(request['schema']), encoding='utf-8')
        command = [
            'codex', 'exec', '--cd', folder, '--sandbox', 'read-only',
            '--ephemeral', '--ignore-user-config', '--skip-git-repo-check',
            '--model', MODEL, '-c', 'model_reasoning_effort="low"',
            '--output-schema', str(schema), '--output-last-message', str(output),
            '--json', '-',
        ]
        response = subprocess.run(
            command, input=request['prompt'].encode('utf-8'),
            capture_output=True, cwd=folder, timeout=60,
        )
        if response.returncode or not output.is_file():
            raise RuntimeError('GPT 시연 연결에서 답변을 받지 못했습니다.')
        events = [json.loads(line) for line in response.stdout.splitlines()]
        if not any(event['type'] == 'turn.completed' for event in events):
            raise RuntimeError('GPT 답변이 완료되지 않았습니다.')
        allowed = {'agent_message', 'reasoning', 'error'}
        if any(event.get('item', {}).get('type') not in allowed
               for event in events if event['type'].startswith('item.')):
            raise RuntimeError('GPT가 질문 답변 외 작업을 시도했습니다.')
        return json.loads(output.read_text(encoding='utf-8')), {
            'provider': 'openai-codex', 'model': MODEL,
        }
