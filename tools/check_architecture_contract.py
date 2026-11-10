"""Source/document consistency only: AST and hashes; no tensors, API or fitting."""
import ast,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def inspect(root=ROOT):
    c=json.loads((root/'docs/model-architecture-contract.json').read_text(encoding='utf-8'))
    errors=[];checks=[]
    def require(ok,name):
        checks.append(name)
        if not ok:errors.append(name)
    parsed={}
    for item in c['code']:
        raw=(root/item['path']).read_bytes().replace(b'\r\n',b'\n')
        require(hashlib.sha256(raw).hexdigest()==item['sha256_lf'],'source hash: '+item['path'])
        parsed[item['path']]=ast.parse(raw.decode('utf-8'))
    def cls(file,name):return next(n for n in parsed['research_code/'+file+'.py'].body if isinstance(n,ast.ClassDef) and n.name==name)
    def init(node):return next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    def linear_values(node):
        return [(ast.literal_eval(n.args[0]),ast.literal_eval(n.args[1])) for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Linear' and all(isinstance(a,ast.Constant) for a in n.args[:2])]
    require(linear_values(init(cls('forecast_transfer','WeeklyEncoder')))==[(420,256),(256,64)],'weekly encoder420->256->64')
    for key,file,name in [('H13','rolling_inventory_model','Forecaster'),('H22','large_inventory_model','LargeForecaster')]:
        m=c['models'][key];tree=init(cls(file,name));linears=linear_values(tree)
        require(tuple(m['own_projection']) in linears,key+' own projection')
        require(tuple(m['neighbor_projection']) in linears,key+' neighbor projection')
        require(tuple(m['rate_decoder']) in linears,key+' rate decoder')
        repeat=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='range']
        require(any(n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==m['context_hidden_layers'] for n in repeat),key+' context depth')
    f=c['feature_policy']
    require(sum(f[k] for k in ('historical_own_width','calendar_width','phase_width','weather_width','station_embedding_width','recent_own_width'))==f['own_concat_width'],'own feature partition')
    require(sum(f[k] for k in ('historical_neighbor_width','station_embedding_width','recent_neighbor_width'))==f['neighbor_concat_width'],'neighbor feature partition')
    require(f['frozen'] is True and f['new_sources'] is False,'feature freeze')
    for name,m in c['models'].items():
        require(m['active_two_seed_parameters']==2*(m['gcn_parameters_per_seed']+m['flow_parameters_per_seed']),name+' declared ensemble size')
        width=m['context_width'] if m['flow_condition_cache'] is None else m['flow_condition_cache'][-1]
        require(m['velocity_decoder'][0]==width+2+2+4,name+' velocity conditioning width')
    require(c['assessment']['new_score_computed'] is False,'no new accuracy claim')
    return {'checks':len(checks),'failures':errors,'kind':'static source/document consistency; not model inference'}

if __name__=='__main__':
    result=inspect();print(json.dumps(result,ensure_ascii=False));raise SystemExit(bool(result['failures']))
