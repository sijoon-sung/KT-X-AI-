"""Local web display and external-request relay; remote facts and validation."""
import argparse,base64,json,subprocess,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from antigravity_transport import generate

def remote(config,value):
    command=subprocess.list2cmdline([config['remote_python'],'-X','utf8',config['remote_repo']+'/runtime/external_assistant.py','--config',config['remote_config']])
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',config.get('ssh_host','dislab'),command],input=json.dumps(value,ensure_ascii=True).encode('ascii'),capture_output=True,timeout=90,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:value=json.loads(r.stdout.decode('utf-8-sig'))
    except (ValueError,UnicodeError):raise RuntimeError('계산 서버 연결을 확인해 주세요.')
    if r.returncode or not value.get('ok'):raise RuntimeError(value.get('error','자료 연결을 확인해 주세요.'))
    return value['result']

def ask(config,question):
    request=remote(config,{'action':'prepare','question':question})
    candidate,metadata=generate(request,config['relay_workspace'],config.get('agy','agy'))
    return remote(config,{'action':'complete','ticket':request['ticket'],'candidate':candidate,'metadata':metadata})

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--port',type=int,default=8773);a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));static=Path(__file__).parent/'assistant_web';dispatch_static=Path(__file__).resolve().parents[1]/'research_code/forecast_web';lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,value,kind='application/json'):
            data=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' https://unpkg.com; style-src 'self' 'unsafe-inline' https://unpkg.com; img-src 'self' data: https://*.tile.openstreetmap.org; connect-src 'self' https://*.tile.openstreetmap.org; frame-ancestors 'none'; base-uri 'none'");self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path in ('/api/health','/api/context'):return self.send(200,remote(config,{'action':path.rsplit('/',1)[-1]}))
                if path=='/api/dispatch/info':return self.send(200,remote(config,{'action':'dispatch_info'}))
                planning={'/dispatch':('dispatch.html','text/html'),'/dispatch/app.js':('dispatch.js','text/javascript'),'/dispatch/style.css':('dispatch.css','text/css')}
                if path in planning:
                    name,kind=planning[path];return self.send(200,(dispatch_static/name).read_bytes(),kind)
                files={'/':('index.html','text/html'),'/assistant':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}
                if path not in files:return self.send(404,{'error':'찾을 수 없습니다.'})
                name,kind=files[path];return self.send(200,(static/name).read_bytes(),kind)
            except Exception:self.send(503,{'error':'자료 연결을 확인해 주세요.'})
        def do_POST(self):
            if self.path not in ('/api/ask','/api/dispatch/optimize'):return self.send(404,{'error':'찾을 수 없습니다.'})
            if not lock.acquire(blocking=False):return self.send(429,{'error':'앞선 답변을 마무리하고 있어요. 잠시 후 다시 보내 주세요.'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=32000:raise ValueError('입력을 확인해 주세요.')
                value=json.loads(self.rfile.read(n))
                if self.path=='/api/dispatch/optimize':
                    result=remote(config,{'action':'dispatch_optimize','body':value});return self.send(result['status'],result['body'])
                question=value.get('question') if isinstance(value,dict) else None
                if not isinstance(question,str) or not 1<=len(question.strip())<=400:raise ValueError('질문은 400자 이내로 적어 주세요.')
                self.send(200,ask(config,question))
            except ValueError:self.send(422,{'error':'질문을 확인해 주세요.'})
            except Exception:self.send(503,{'error':'외부 답변 연결을 확인해 주세요. 원래 계획은 자료에서 볼 수 있습니다.'})
            finally:lock.release()
    print(f'Tashu external assistant: http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
