"""Run the unchanged H21 RPC on the compute server, using a private stream."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research_code'))
import bulk_dispatch

if __name__ == '__main__':
    try:
        folder = Path(os.environ['TASHU_STREAM_DIR'])
        if not folder.is_dir():
            raise ValueError('Stream directory unavailable.')
        response = bulk_dispatch.rpc(folder, json.load(sys.stdin))
    except (ValueError, KeyError, TypeError):
        response = dict(status=422, body=dict(error='관측 폴더와 입력 조건을 확인하세요.'))
    except Exception:
        response = dict(status=503, body=dict(error='계획 계산을 완료하지 못했습니다.'))
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
