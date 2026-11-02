"""Loopback static display and private stdin bridge; all planning stays remote."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit,parse_qs


def handler(info,bridge):
    static=Path(__file__).resolve().parent/'forecast_web'
    class Display(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,raw,kind='application/json'):
            self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def error(self,status,message):self.send(status,json.dumps(dict(error=message),ensure_ascii=False).encode())
        def remote(self,request):
            r=subprocess.run([sys.executable,str(bridge)],input=json.dumps(request,ensure_ascii=False).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=75,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if r.returncode:raise RuntimeError('계산 장비에 연결하지 못했습니다.')
            return json.loads(r.stdout.decode('utf-8-sig'))
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/api/dispatch/info':
                if parse_qs(urlsplit(self.path).query).get('fresh')==['1']:
                    try:
                        result=self.remote(dict(path=path));return self.send(result['status'],json.dumps(result['body'],ensure_ascii=False,allow_nan=False).encode())
                    except Exception:return self.error(503,'최신 관측 연결을 다시 확인하고 있습니다.')
                if not info.is_file():return self.error(503,'관측 자료를 불러올 수 없습니다.')
                return self.send(200,info.read_bytes())
            files={'/':('dispatch.html','text/html'),'/results':('dispatch.html','text/html'),'/dispatch':('dispatch.html','text/html'),
                   '/dispatch/app.js':('dispatch.js','text/javascript'),'/dispatch/style.css':('dispatch.css','text/css')}
            if path not in files:return self.error(404,'찾을 수 없습니다.')
            name,kind=files[path];self.send(200,(static/name).read_bytes(),kind)
        def do_POST(self):
            if self.path!='/api/dispatch/optimize':return self.error(404,'찾을 수 없습니다.')
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<32000:raise ValueError('입력을 확인하세요.')
                value=json.loads(self.rfile.read(size));result=self.remote(dict(path=self.path,body=value));self.send(result['status'],json.dumps(result['body'],ensure_ascii=False,allow_nan=False).encode())
            except ValueError:return self.error(422,'입력 조건을 확인하세요.')
            except Exception:return self.error(503,'경로를 계산하지 못했습니다. 이전 경로는 새 조건의 결과가 아닙니다.')
    return Display


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--info',required=True);p.add_argument('--bridge',required=True);p.add_argument('--port',type=int,default=8770)
    a=p.parse_args();server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(Path(a.info),Path(a.bridge)))
    print(f'Local: http://127.0.0.1:{a.port}/dispatch',flush=True);server.serve_forever()
