"""Build actual trained H22 weights/runtime on the compute host, no API data."""
import argparse,ast,hashlib,importlib.metadata,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_code'))
from large_inventory_model import digest,LargeForecaster,CachedFlow
from large_inventory_fit import write
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    fit=Path(a.fit);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    bundle=out/'tashu-h22-100mb-v1.0.1';(bundle/'model').mkdir(parents=True);(bundle/'code').mkdir()
    lock=json.loads((fit/'model-lock.json').read_text(encoding='utf-8'));sha=digest(fit/'model-lock.json')
    if lock['hypothesis_id']!='H-0022' or lock['api_opened'] is not False:raise ValueError('Wrong lock.')
    expected={'context.npz','time-features.npz','large-11.pt','large-29.pt'}
    if {x['id'] for x in lock['artifacts']}!=expected:raise ValueError('Unexpected artifacts.')
    shutil.copy2(fit/'model-lock.json',bundle/'model/model-lock.json')
    for item in lock['artifacts']:
        if digest(fit/item['id'])!=item['sha256']:raise ValueError('Artifact changed.')
        shutil.copy2(fit/item['id'],bundle/'model'/item['id'])
    queue=['large_inventory_model','rolling_inventory_predict','live_api_service','forecast_transfer_data'];selected=set();code={x['id']:x['sha256'] for x in lock['code']}
    while queue:
        module=queue.pop()
        if module in selected:continue
        selected.add(module);file=ROOT/'research_code'/(module+'.py');raw=file.read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(raw).hexdigest()!=code[file.name]:raise ValueError('Frozen numerical code changed: '+file.name)
        (bundle/'code'/file.name).write_bytes(raw)
        for node in ast.walk(ast.parse(raw.decode())):
            names=[x.name.split('.')[0] for x in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            queue.extend(x for x in names if (ROOT/'research_code'/(x+'.py')).is_file())
    shutil.copy2(ROOT/'runtime/run.py',bundle/'run.py')
    (bundle/'large_run.py').write_text((ROOT/'runtime/large_run.py').read_text(encoding='utf-8').replace('BUILD_REPLACES_WITH_LOCK_SHA',sha),encoding='utf-8')
    shutil.copy2(ROOT/'runtime/requirements.txt',bundle/'requirements.txt')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    def linked_document(path):
        def replace(match):
            target=match.group(1)
            if target.startswith(('https://','http://','#')):return match.group(0)
            relative=target if target.startswith('docs/') else 'docs/'+target
            if not (ROOT/relative.split('#')[0]).is_file():raise ValueError('Missing package documentation link: '+relative)
            return '](https://github.com/sijoon-sung/KT-X-AI-/blob/'+source+'/'+relative+')'
        return re.sub(r'\]\(([^)]+)\)',replace,path.read_text(encoding='utf-8'))
    (bundle/'MODEL_CARD.md').write_text(linked_document(ROOT/'docs/model-card-h22.md'),encoding='utf-8')
    (bundle/'README.md').write_text(linked_document(ROOT/'docs/quickstart-h22.md'),encoding='utf-8')
    weights=[];parameters=0
    for seed in (11,29):
        saved=torch.load(fit/f'large-{seed}.pt',map_location='cpu',weights_only=True);gcn=LargeForecaster();flow=CachedFlow()
        gcn.load_state_dict(saved['gcn'],strict=True);flow.load_state_dict(saved['flow'],strict=True)
        count=lambda x:sum(p.numel() for p in x.parameters())
        tensor_bytes=sum(v.numel()*v.element_size() for key in ('gcn','flow') for v in saved[key].values())
        parameters+=count(gcn)+count(flow)
        weights.append({'name':f'large-{seed}.pt','gcn_parameters':count(gcn),'flow_parameters':count(flow),'file_bytes':(fit/f'large-{seed}.pt').stat().st_size,'tensor_bytes':tensor_bytes,'sha256':digest(fit/f'large-{seed}.pt')})
        shutil.copy2(fit/f'large-{seed}.pt',out/f'large-{seed}.pt')
    total=sum(x['tensor_bytes'] for x in weights)
    if not 90_000_000<=total<=110_000_000:raise ValueError('Deployment weights outside frozen size range.')
    report={'hypothesis_id':'H-0022','package_version':'1.0.1','model_lock_sha256':sha,'source_commit':source,'all_loaded_parameters':parameters,'checkpoint_tensor_bytes':total,'checkpoint_file_bytes':sum(x['file_bytes'] for x in weights),'runtime_models':weights,'context_file_bytes':(fit/'context.npz').stat().st_size,'environment':{k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','pandas')},'api_training':False,'raw_api_included':False,'raw_trip_events_included':False,'credentials_included':False}
    write(bundle/'model-info.json',report)
    immutable=lambda f:f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'
    write(bundle/'package-manifest.json',{'model_lock_sha256':sha,'files':[{'path':f.relative_to(bundle).as_posix(),'sha256':digest(f),'bytes':f.stat().st_size} for f in sorted(bundle.rglob('*')) if immutable(f)]})
    archive=out/'tashu-h22-100mb-v1.0.1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in sorted(bundle.rglob('*')):
            if immutable(f):z.write(f,f.relative_to(out).as_posix())
    report['archive']={'name':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)}
    (out/(archive.name+'.sha256')).write_text(report['archive']['sha256']+'  '+archive.name+'\n',encoding='ascii')
    write(out/'model-publication.json',report);print(json.dumps(report))

if __name__=='__main__':main()
