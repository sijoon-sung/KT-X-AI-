"""Loopback-only viewer for one actual route computation and hosted answer."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from antigravity_transport import generate
from gpt_demo_transport import generate as generate_gpt


def main():
    parser = argparse.ArgumentParser()
    for key in ('config','viewer','capture-log'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--port',type=int,default=8778)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    provider = config.get('provider', 'antigravity')
    if provider not in ('antigravity', 'openai-codex'):
        parser.error('Unknown answer provider')
    viewer = args.viewer.read_text(encoding='utf-8')
    saved = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',viewer,re.S)[1])
    empty = {'plans':{'cases':[]},'review':{'selected':[],'fleet':[]},'network':saved['network']}
    page = viewer.replace('<body data-presentation="1">',
                          f'<body data-presentation="1" data-live="1" data-provider="{provider}">')
    page = re.sub(r'(<script id="payload" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(empty,separators=(',',':'))+m[2],page,flags=re.S)
    lock = threading.Lock()
    state = None

    def remote(value):
        command = subprocess.list2cmdline([config['remote_python'],'-X','utf8',config['remote_worker'],'--config',config['worker_config']])
        proc = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',config['ssh_host'],command],
                              input=json.dumps(value,ensure_ascii=True).encode(),capture_output=True,timeout=70)
        if proc.returncode:
            print(proc.stderr.decode('utf-8',errors='replace')[-1600:],flush=True)
            raise RuntimeError('계산 서버에서 요청을 마치지 못했습니다.')
        return json.loads(proc.stdout.decode('utf-8-sig'))

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, kind='application/json'):
            data = value.encode() if isinstance(value, str) else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':return self.send(200,page,'text/html')
            return self.send(404,{'error':'페이지를 찾지 못했습니다.'})

        def do_POST(self):
            nonlocal state
            # A page on another origin cannot submit a JSON request to this relay.
            if self.headers.get('Origin') != f'http://127.0.0.1:{args.port}' or self.headers.get_content_type()!='application/json':
                return self.send(403,{'error':'시연 화면에서 요청해 주세요.'})
            if not lock.acquire(blocking=False):return self.send(429,{'error':'앞선 작업을 마치는 중입니다.'})
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0<size<=4096:raise ValueError('요청 내용을 확인해 주세요.')
                data = json.loads(self.rfile.read(size))
                started = time.monotonic()
                if self.path == '/api/product-plan':
                    state = None
                    state = remote({'action':'plan'})
                    result={'plans':{'cases':state['cases']},'review':state['review']}
                    record={'event':'plan','mode':state['mode'],'cases':[{'id':c['id'],'moved_bikes':c['moved_bikes']} for c in state['cases']]}
                elif self.path == '/api/product-ask':
                    if state is None:raise ValueError('먼저 경로를 계산해 주세요.')
                    question = data.get('question')
                    if not isinstance(question,str) or not 0<len(question.strip())<=160:raise ValueError('질문을 확인해 주세요.')
                    case = next((c for c in state['cases'] if c['id']==data.get('case')),None)
                    if case is None:raise ValueError('권역의 작업표를 열어 주세요.')
                    request = remote({'action':'prepare','case':case,'question':question})
                    if provider == 'openai-codex':
                        response, metadata = generate_gpt(request, config['relay_workspace'])
                    else:
                        response, metadata = generate(request, config['relay_workspace'], config['agy'])
                    result = remote({'action':'complete','case':case,'candidate':response})
                    result['provider'] = provider
                    if provider == 'openai-codex':
                        result['model'] = metadata['model']
                    record=dict(event='answer',case=case['id'],question=question,transport=metadata,**result)
                else:return self.send(404,{'error':'요청을 찾지 못했습니다.'})
                record['elapsed_seconds']=round(time.monotonic()-started,3)
                with args.capture_log.open('a',encoding='utf-8') as log:log.write(json.dumps(record,ensure_ascii=False)+'\n')
                self.send(200,result)
            except ValueError as exc:self.send(422,{'error':str(exc)})
            except Exception as exc:
                print(type(exc).__name__,str(exc),flush=True)
                self.send(503,{'error':'요청을 마치지 못했습니다. 연결을 확인해 주세요.'})
            finally:lock.release()

    print(f'Product demo: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__ == '__main__':
    main()
