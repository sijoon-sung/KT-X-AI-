"""Local display-only HTTP proxy. All answers and context run on the SSH host."""
import argparse,base64,json,subprocess
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

def main():
    p=argparse.ArgumentParser();p.add_argument('--ssh-host',default='dislab');p.add_argument('--remote-port',type=int,default=8782);p.add_argument('--port',type=int,default=8772);a=p.parse_args()
    if not 1<=a.remote_port<=65535 or not 1<=a.port<=65535 or a.ssh_host.startswith('-'):p.error('Invalid host/port')
    static=Path(__file__).parent/'assistant_web'
    def forward(path,body=None):
        script="$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        if body is not None:
            encoded=base64.b64encode(body).decode()
            script+="$body=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+encoded+"'));"
        script+="$reply=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:"+str(a.remote_port)+path+"' -TimeoutSec 60"
        if body is not None:script+=" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))"
        script+=";[Console]::Write($reply.Content)"
        encoded=base64.b64encode(script.encode('utf-16le')).decode()
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',a.ssh_host,'powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',encoded],capture_output=True,timeout=85)
        if result.returncode:raise RuntimeError('Remote assistant unavailable')
        value=json.loads(result.stdout.decode('utf-8-sig'));return json.dumps(value,ensure_ascii=False).encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data,kind='application/json'):
            self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; frame-ancestors 'none'; base-uri 'none'");self.end_headers();self.wfile.write(data)
        def fail(self):self.send(503,json.dumps({'error':'원격 도우미 연결을 확인해 주세요.'},ensure_ascii=False).encode())
        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path in ('/api/health','/api/context'):return self.send(200,forward(path))
                files={'/':('index.html','text/html'),'/assistant':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}
                if path not in files:return self.send(404,b'{}')
                name,kind=files[path];return self.send(200,(static/name).read_bytes(),kind)
            except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired):self.fail()
        def do_POST(self):
            if self.path!='/api/ask':return self.send(404,b'{}')
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=8192:return self.send(422,b'{}')
                body=self.rfile.read(n);json.loads(body);return self.send(200,forward('/api/ask',body))
            except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired):self.fail()
    print(f'Display-only: http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
