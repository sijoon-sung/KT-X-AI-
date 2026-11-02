"""Serve the existing dispatch UI and unchanged H21 calculation on dislab."""
import argparse
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research_code'))
import dispatch_preview

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream-dir', required=True)
    p.add_argument('--port', type=int, default=8770)
    a = p.parse_args()
    folder = Path(a.stream_dir).resolve()
    if not (folder / 'status.json').is_file():
        p.error('Existing private observation stream status.json required.')
    os.environ['TASHU_STREAM_DIR'] = str(folder)
    base = dispatch_preview.handler(folder / 'display-info-unused.json', ROOT / 'runtime/dispatch_rpc.py')
    class CurrentMetadata(base):
        def do_GET(self):
            if urlsplit(self.path).path == '/api/dispatch/info':
                self.path = '/api/dispatch/info?fresh=1'
            super().do_GET()
    print(f'Local: http://127.0.0.1:{a.port}/dispatch', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), CurrentMetadata).serve_forever()
