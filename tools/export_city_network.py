"""Export a fixed representative view of the historical GCN for video rendering."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--context',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    boundary=a.out/'boundary.osm.json'
    url='https://api.openstreetmap.org/api/0.6/relation/2349984/full.json'
    if not boundary.exists():
        with urlopen(Request(url,headers={'User-Agent':'TashuMapStory/1.0 (+https://github.com/sijoon-sung/KT-X-AI-)'}),timeout=25) as response:
            raw=response.read(10_000_001)
        if len(raw)>10_000_000:raise ValueError('Boundary response exceeded limit')
        boundary.write_bytes(raw)
    boundary_data=json.loads(boundary.read_bytes())['elements']
    relation=next(e for e in boundary_data if e['type']=='relation' and e['id']==2349984)
    if relation['tags'].get('name')!='대전광역시':raise ValueError('Unexpected administrative boundary')
    member_ids={m['ref'] for m in relation['members'] if m['type']=='way' and m['role']=='outer'}
    points={e['id']:[e['lon'],e['lat']] for e in boundary_data if e['type']=='node'}
    outline=[[points[n] for n in e['nodes']] for e in boundary_data if e['type']=='way' and e['id'] in member_ids]
    stations={s['id']:s for s in json.loads(a.state.read_text(encoding='utf-8'))['stations']}
    with np.load(a.context,allow_pickle=False) as data:
        ids=data['station_ids'].astype(str).tolist();indptr=data['graph_indptr'];indices=data['graph_indices'];weights=data['graph_data']
        pairs={}
        for i,sid in enumerate(ids):
            if sid not in stations:continue
            for j,value in zip(indices[indptr[i]:indptr[i+1]],weights[indptr[i]:indptr[i+1]]):
                other=ids[j]
                if other not in stations or sid==other or value<=0:continue
                pair=tuple(sorted((sid,other)));pairs[pair]=max(pairs.get(pair,0),float(value))
    points={sid:[s['lon'],s['lat']] for sid,s in stations.items()}
    cells={}
    for pair,value in pairs.items():
        first,second=(points[k] for k in pair)
        grid=tuple(int((a+b)*80) for a,b in zip(first,second))
        cells.setdefault(grid,[]).append((pair,value))
    for items in cells.values():items.sort(key=lambda row:-row[1])
    visible=[]
    for rank in range(3):
        for key in sorted(cells):
            if rank<len(cells[key]):visible.append(cells[key][rank][0])
    payload=dict(boundary=outline,boundary_source=url,stations=list(points.values()),
                 edges=[[points[a],points[b]] for a,b in visible],graph_edge_count=len(pairs))
    (a.out/'city-network.json').write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
    print(json.dumps(dict(stations=len(points),edges=len(pairs),displayed=len(visible),boundary_ways=len(outline))))


if __name__=='__main__':main()
