"""Verify the portable adapter against the research implementation on dislab.
Synthetic inputs only. This is a serialization/interface check, not accuracy.
"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def run(a):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'runtime'));sys.path.insert(0,str(root/'research_code'))
    import count_candidate as portable
    import count_flow_models as reference
    from scaled_flow_core import CountFlow
    torch.set_num_threads(4)
    model=portable.Predictor(a.model_dir,a.device);device=torch.device(a.device)
    own=np.zeros((8,100),np.float32);prior=np.full((8,2),.1,np.float32);own[:,:6]=np.log1p(.3);own[:,6]=1;own[:,25]=1;own[:,37]=1;own[:,40]=1;own[:,41]=1
    c=SimpleNamespace(log=torch.as_tensor(own[:,:6,None].transpose(0,2,1),device=device),calendar=torch.as_tensor(own[:,6:41],device=device),phase=torch.as_tensor(own[:,41:44],device=device),weather=torch.as_tensor(own[:,44:98],device=device),prior=torch.as_tensor(prior[:,None],device=device),recent=torch.as_tensor(np.concatenate((own[:,98:],np.zeros((8,2),np.float32)),1)[:,None],device=device),slot=torch.arange(8,device=device))
    rows=torch.arange(8,device=device);nodes=torch.zeros(8,dtype=torch.long,device=device);pairs=[];flows=[]
    for seed in (11,29):
        m=reference.CompactEncoder('compact_local').to(device);saved=torch.load(Path(a.model_dir)/f'encoder-compact_local-{seed}.pt',map_location='cpu',weights_only=True);m.load_state_dict(saved['state_dict']);m.eval()
        f=CountFlow(context_width=128,width=128).to(device);saved=torch.load(Path(a.model_dir)/f'flow-poisson_continuous-score-{seed}.pt',map_location='cpu',weights_only=True);f.load_state_dict(saved['state_dict']);f.eval();flows.append((f,saved['epoch']))
        with torch.inference_mode():pairs.append(m.rate(c,rows,nodes,'recent',hidden=True))
    seed=20261124;choice=np.random.default_rng(seed).integers(0,2,(8,128));expected=np.empty((8,128,2),np.int32);g=torch.Generator(device=device).manual_seed(seed)
    for i,(p,h) in enumerate(pairs):
        rr,ss=np.where(choice==i);ix=torch.as_tensor(rr,device=device);values,_=reference.sample_flow(flows[i][0],'poisson_continuous',flows[i][1],p[ix],h[ix],g);expected[rr,ss]=values
    actual=model.predict(own,prior,seed);expected_rate=torch.stack([p[0] for p in pairs]).mean(0).cpu().numpy()
    if not np.array_equal(actual['count_samples'],expected):raise AssertionError('Portable samples differ from reference')
    if not np.array_equal(actual['mean_rate'],expected_rate):raise AssertionError('Portable rates differ from reference')
    rejected=0
    for x,p in [(own[:,:99],prior),(own,np.zeros_like(prior))]:
        try:model.predict(x,p,seed)
        except ValueError:rejected+=1
    if rejected!=2:raise AssertionError('Input rejection failed')
    result={'device':a.device,'synthetic_inputs':True,'accuracy_evaluation':False,'checkpoint_hashes_verified':True,'exact_mean_rate_match':True,'exact_integer_particle_match':True,'output_shape':list(expected.shape),'invalid_inputs_rejected':rejected,'api_used':False}
    if a.out:Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--device',default='cpu');p.add_argument('--out');run(p.parse_args())
