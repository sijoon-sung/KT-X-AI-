"""Loopback display only: serve an already computed private evidence bundle."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


def handler(bundle):
    static = Path(__file__).resolve().parent/'forecast_web'
    class Display(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = urlsplit(self.path).path
            routes = {'/': ('results.html', 'text/html'), '/results': ('results.html', 'text/html'),
                      '/results/': ('results.html', 'text/html'), '/results/app.js': ('results.js', 'text/javascript'),
                      '/results/style.css': ('results.css', 'text/css')}
            status = 200
            if path == '/api/results':
                kind = 'application/json'
                if bundle.is_file():
                    raw = bundle.read_bytes()
                else:
                    status = 503
                    raw = json.dumps({'error':'Verified private result bundle is unavailable.'}).encode()
            elif path in routes:
                filename, kind = routes[path]
                raw = (static/filename).read_bytes()
            else:
                status, kind, raw = 404, 'text/plain', b'Not found'
            self.send_response(status)
            self.send_header('Content-Type', kind+'; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)
    return Display


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--port',type=int,default=8769)
    a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(Path(a.bundle).resolve()))
    print(f'Local: http://127.0.0.1:{a.port}/results',flush=True)
    server.serve_forever()
