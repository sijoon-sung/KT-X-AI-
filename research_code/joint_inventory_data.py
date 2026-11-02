"""H14 historical tensor/region audit. No inventory API loader or fitting."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
import historical_calendar as h
import rolling_inventory_data as d
import cross_city_weather as previous

ROOT=Path(__file__).resolve().parents[1]
def regions(graph,ids,size=16):
    w=(graph+graph.T).toarray();np.fill_diagonal(w,0);degree=w.sum(1);remaining=np.ones(len(ids),bool);out=[]
    while remaining.any():
        candidates=np.flatnonzero(remaining);seed=candidates[np.lexsort((ids[candidates],-degree[candidates]))[0]];group=[int(seed)];remaining[seed]=False
        while remaining.any() and len(group)<size:
            candidates=np.flatnonzero(remaining);affinity=w[candidates][:,group].sum(1)
            order=np.lexsort((ids[candidates],-degree[candidates],-affinity));node=int(candidates[order[0]]);group.append(node);remaining[node]=False
        out.append(group)
    return out

def pairs(graph,n):
    coo=(graph+graph.T).tocoo();edges=sorted(set((min(int(i),int(j)),max(int(i),int(j))) for i,j in zip(coo.row,coo.col) if i!=j));rng=np.random.default_rng(20261104)
    count=min(2048,len(edges));ix=rng.choice(len(edges),count,replace=False);chosen=[edges[i] for i in sorted(ix)];known=set(edges);other=set()
    while len(other)<min(2048,n*(n-1)//2-len(known)):
        i,j=sorted(rng.integers(0,n,2).tolist())
        if i!=j and (i,j) not in known:other.add((i,j))
    all_pairs=chosen+sorted(other);weights=np.r_[np.full(len(chosen),.5/max(1,len(chosen))),np.full(len(other),.5/max(1,len(other)))]
    return np.asarray(all_pairs,dtype='int64'),weights.astype('float32')

def prepare(a):
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True);parent=Path(a.parent_fit)
    if (out/'audit.json').exists():raise ValueError('Preserve existing audit output.')
    record=json.loads((ROOT/'research/datasets/tashu-dependence-flow-H0014-design-v1.json').read_text(encoding='utf-8'));anchor=record['fixed_historical_anchor']
    for item in anchor['artifacts']:
        if h.digest(parent/item['id'])!=item['sha256']:raise ValueError('Historical parent artifact hash mismatch.')
        shutil.copyfile(parent/item['id'],out/item['id'])
    source=next(s for s in record['historical_sources'] if s['id']=='historical-trips-20260331.zip')
    if h.digest(a.archive)!=source['sha256']:raise ValueError('Historical archive bytes differ.')
    with np.load(out/'context.npz',allow_pickle=False) as f:
        ids=f['station_ids'].copy();first=f['first'].copy();n=len(ids);graph=sparse.csr_matrix((f['graph_data'],f['graph_indices'],f['graph_indptr']),shape=(n,n))
    events,source_ids,months,quality=h.read_events(Path(a.archive),previous.historical_source_catalog());panel,counts_audit=d.aggregate(events,source_ids,ids,months);del events,source_ids
    with np.load(out/'time-features.npz',allow_pickle=False) as f:
        if not np.array_equal(f['times'],panel['times']) or not np.array_equal(f['valid'],panel['valid']):raise ValueError('Rebuilt historical support differs from parent.')
    np.save(out/'counts.npy',panel['counts'],allow_pickle=False)
    group=regions(graph,ids);edge,weight=pairs(graph,n);train=panel['valid']&(panel['times']<d.TRAIN_END);total=np.zeros(n);exposure=np.zeros(n)
    for begin in range(0,len(panel['times']),256):
        end=min(begin+256,len(panel['times']));valid=train[begin:end,None]&(panel['times'][begin:end,None]>=first)
        net=panel['counts'][begin:end,:,1].astype(float)-panel['counts'][begin:end,:,0].astype(float);total+=(net**2*valid).sum(0);exposure+=valid.sum(0)
    scale=np.maximum(1,np.sqrt(total/np.maximum(exposure,1))).astype('float32')
    group_id=np.empty(n,dtype='int64')
    for i,g in enumerate(group):group_id[g]=i
    np.savez_compressed(out/'dependence-layout.npz',group=group_id,edge=edge,edge_weight=weight,scale=scale)
    rng=np.random.default_rng(20261104);day=(panel['times']+32400)//86400;selection=[];aux=[]
    for role,start,end,per_day,dest in [('validation',d.TRAIN_END,d.VALID_END,2,selection),('auxiliary',d.VALID_END,d.END,1,aux)]:
        eligible=panel['valid']&(panel['times']>=start)&(panel['times']<end)
        for dd in np.unique(day[eligible]):
            choices=np.flatnonzero(eligible&(day==dd));dest.extend(sorted(rng.choice(choices,min(per_day,len(choices)),replace=False).tolist()))
    np.savez_compressed(out/'fixed-evaluation-issues.npz',validation=np.array(selection),auxiliary=np.array(aux))
    report=dict(hypothesis_id='H-0014',audit_scope='historical-only source/support/layout; no model comparison; no API opened',source=quality,panel=counts_audit,stations=n,regions=len(group),region_sizes=[len(g) for g in group],pairs=len(edge),validation_issues=len(selection),auxiliary_issues=len(aux),selection_seed=20261104,
        training_issue_times=int(train.sum()),training_node_exposures=int(exposure.sum()),files=[dict(id=p.name,bytes=p.stat().st_size,sha256=h.digest(p)) for p in sorted(out.iterdir()) if p.is_file()])
    h.write(out/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('files','source','region_sizes')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--parent-fit',required=True);p.add_argument('--out',required=True);prepare(p.parse_args())
