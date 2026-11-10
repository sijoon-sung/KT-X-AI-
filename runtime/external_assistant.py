"""Prepare and verify hosted-LLM answers on the compute server; no model fitting."""
import argparse,base64,hashlib,hmac,json,os,secrets,time
from pathlib import Path
from urllib.request import Request,urlopen
from assistant_context import evidence,focus,load_context,validate_answer,fallback

MODEL='gemini-3.8-flash-low'
PROMPT='''타슈 운영 담당자에게 쉬운 한국어로 짧게 답하세요. 아래 사실만 사용합니다.
질문에 맞는 수량·대여소 이름을 그대로 복사하세요. 현재 재고와 실을 대수를 혼동하지 마세요.
경로를 물으면 제공된 모든 방문을 순서대로 한 줄씩 적고,각 방문의 싣기/내려놓기 대수를 적으세요.
전체 계획을 요청하면 먼저 총 이동 대수·거리·예상 시간을 쓰고, 모든 방문지의 싣기/내려놓기 대수를 순서대로 적으세요.
자료 시각은 질문할 때만 답하세요. 기록과 예시를 현재 실시간이라고 말하지 마세요.
실제로 자전거를 못 빌린 인원은 알 수 없다고 답하세요. 새 수량·경로·순위를 계산하지 마세요.
답변에 쓴 사실의 ID를 evidence_ids에 적으세요. 파일·명령·웹·MCP·다른 도구는 사용하지 마세요.
JSON의 answer는 자연스러운 한국어이고,기술 설명은 질문할 때만 합니다.'''

def encode(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()

class Gateway:
    def __init__(self,config):
        self.config=config;path=Path(config['ticket_key']);path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with path.open('xb') as f:f.write(secrets.token_bytes(32))
        except FileExistsError:pass
        self.key=path.read_bytes()
        if len(self.key)!=32:raise ValueError('Invalid private ticket key')
    def raw(self):
        s=self.config['source']
        if s.get('stream'):
            from operator_pipeline import context
            return context(s)
        return load_context(s.get('context'),s.get('forecast'),s.get('plan'))
    def context(self):return evidence(self.raw())
    def prepare(self,question):
        if not isinstance(question,str) or not 1<=len(question.strip())<=400:raise ValueError('질문은 400자 이내로 적어 주세요.')
        question=question.strip();pack=focus(evidence(self.raw(),question),question)
        body={'question':question,'pack':pack,'expires':time.time()+240,'nonce':secrets.token_hex(16)}
        payload=base64.urlsafe_b64encode(encode(body)).decode();sig=hmac.new(self.key,payload.encode(),hashlib.sha256).hexdigest()
        ids=[x['id'] for x in pack['facts']]
        schema={'type':'object','additionalProperties':False,'properties':{'answer':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string','enum':ids}}},'required':['answer','evidence_ids']}
        facts='\n'.join('['+x['id']+'] '+x['text'] for x in pack['facts'])
        return {'ticket':payload+'.'+sig,'prompt':PROMPT+'\n사실:\n'+facts+'\n질문: '+question,'schema':schema,'model':self.config.get('model',MODEL),'context_id':pack['context_id']}
    def complete(self,ticket,candidate,metadata=None):
        payload,sig=ticket.rsplit('.',1)
        expected=hmac.new(self.key,payload.encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig,expected):raise ValueError('잘못된 요청입니다.')
        body=json.loads(base64.urlsafe_b64decode(payload));pack=body['pack'];metadata=metadata or {}
        if time.time()>body['expires']:raise ValueError('답변 요청이 만료됐습니다. 다시 물어보세요.')
        if self.context()['context_id']!=pack['context_id']:raise ValueError('자료가 바뀌었습니다. 새 계획으로 다시 물어보세요.')
        try:
            result=validate_answer(candidate,pack);engine='external';reason=None
        except (ValueError,KeyError,TypeError):
            result=fallback(pack,body['question']);engine='facts';reason='계획과 답변을 맞추지 못해 원래 자료를 보여드립니다.'
        result.update(engine=engine,model=self.config.get('model',MODEL),provider=self.config.get('provider','antigravity'),context_id=pack['context_id'],source_label=pack['label'],fallback_reason=reason)
        result['evidence']=[x['text'] for x in pack['facts'] if x['id'] in result['evidence_ids']]
        result['transport']={k:metadata[k] for k in ('seconds','usage','provider') if k in metadata}
        return result
    def http_answer(self,question):
        r=self.prepare(question);cfg=self.config['http'];url=cfg['url']
        if not(url.startswith('https://') or url.startswith('http://127.0.0.1:') or url.startswith('http://localhost:')):raise ValueError('HTTPS API 주소가 필요합니다.')
        key=os.environ.get(cfg.get('key_env','OPENAI_API_KEY'),'')
        if not key:raise ValueError('서버의 외부 API 키를 설정하세요.')
        payload={'model':self.config['model'],'messages':[{'role':'user','content':r['prompt']}],'response_format':{'type':'json_schema','json_schema':{'name':'tashu_answer','strict':True,'schema':r['schema']}}}
        req=Request(url,data=encode(payload),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        started=time.monotonic()
        with urlopen(req,timeout=60) as response:
            data=response.read(2_000_001)
        if len(data)>2_000_000:raise ValueError('API 응답이 너무 큽니다.')
        value=json.loads(data);candidate=json.loads(value['choices'][0]['message']['content'])
        return self.complete(r['ticket'],candidate,{'seconds':time.monotonic()-started,'usage':value.get('usage',{}),'provider':'external-http'})

def dispatch(gateway,value):
    if not isinstance(value,dict):raise ValueError('잘못된 요청입니다.')
    action=value.get('action')
    if action=='context':return gateway.context()
    if action in ('dispatch_info','dispatch_optimize'):
        from operator_pipeline import info,optimize
        return info(gateway.config['source']) if action=='dispatch_info' else optimize(gateway.config['source'],value.get('body'))
    if action=='health':return {'ready':True,'provider':gateway.config.get('provider','antigravity'),'model':gateway.config.get('model',MODEL),'compute':'remote preparation and validation; hosted language model'}
    if action=='prepare':return gateway.prepare(value.get('question'))
    if action=='complete':return gateway.complete(value['ticket'],value.get('candidate'),value.get('metadata'))
    if action=='ask':return gateway.http_answer(value.get('question'))
    raise ValueError('지원하지 않는 요청입니다.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--port',type=int);a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));gateway=Gateway(config)
    if not a.port:
        import sys
        try:result=dispatch(gateway,json.load(sys.stdin));print(json.dumps({'ok':True,'result':result},ensure_ascii=False))
        except Exception as e:print(json.dumps({'ok':False,'error':str(e) if isinstance(e,ValueError) else '외부 답변 연결을 확인하세요.'},ensure_ascii=False));raise SystemExit(2)
        return
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            action={'/api/health':'health','/api/context':'context'}.get(self.path)
            self.respond({'action':action},200 if action else 404)
        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length',0))
                if self.path!='/api/ask' or not 0<n<=8192:return self.respond({},422)
                value=json.loads(self.rfile.read(n));value['action']='ask';self.respond(value)
            except Exception:self.respond({},422)
        def respond(self,value,status=200):
            try:result=dispatch(gateway,value) if status==200 else {'error':'요청을 확인하세요.'}
            except Exception:result={'error':'서버 API 설정과 자료를 확인하세요.'};status=503
            data=encode(result);self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
