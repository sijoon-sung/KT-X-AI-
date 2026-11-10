"""Network-only relay via the user's normally authenticated official CLI."""
import json,subprocess,time
from pathlib import Path

SETTINGS={'toolPermission':'strict','permissions':{'deny':[x+'(*)' for x in ('read_file','write_file','command','read_url','execute_url','mcp')]},'notifications':False,'allowNonWorkspaceAccess':False}

def generate(request,cwd,executable='agy'):
    cwd=Path(cwd);cwd.mkdir(parents=True,exist_ok=True)
    args=[executable,'--disable-slash-commands','--settings',json.dumps(SETTINGS),'--output-format','json','--json-schema',json.dumps(request['schema'],ensure_ascii=False),'--model',request['model'],'--print-timeout','90s','-p',request['prompt']]
    started=time.monotonic()
    result=subprocess.run(args,cwd=cwd,stdin=subprocess.DEVNULL,capture_output=True,encoding='utf-8',errors='replace',timeout=100,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('Antigravity 연결 또는 사용 한도를 확인해 주세요.')
    value=json.loads(result.stdout)
    if value.get('status')!='SUCCESS':raise RuntimeError('Antigravity 답변을 받지 못했습니다.')
    candidate=value.get('structured_output')
    if candidate is None:candidate=json.loads(value['response'])
    return candidate,{'provider':'antigravity-official-cli','seconds':time.monotonic()-started,'usage':value.get('usage',{})}
