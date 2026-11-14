"""Export a portable, offline viewer of computed routes and review evidence."""
import argparse
import json
from pathlib import Path


HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>타슈 재배치 시나리오</title><style>
*{box-sizing:border-box}body{margin:0;background:#0d1827;color:#f1f6fd;font:16px/1.55 "Malgun Gothic",system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:32px}h1{margin:0;font-size:clamp(1.6rem,4vw,2.65rem)}p{margin:.5em 0}.muted{color:#abc0d3}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;flex-wrap:wrap}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}button,a.download{font:inherit;color:#ebf6ff;background:#1b3047;border:1px solid #365165;border-radius:8px;padding:8px 13px;cursor:pointer;text-decoration:none}button[aria-pressed=true]{background:#1f534e;border-color:#61e6ce}button:focus-visible,a:focus-visible{outline:3px solid #fff}.grid{display:grid;grid-template-columns:1.7fr 1fr;gap:14px}.panel{background:#14243a;border:1px solid #263d54;border-radius:14px;padding:17px;min-width:0}.map{width:100%;display:block;border-radius:8px;background:#101c2b;min-height:290px}.metric{color:#61e6ce;font-size:2.1rem;font-weight:750}.case{padding:12px 0;border-bottom:1px solid #365165}.case:last-child{border:0}.case h3{margin:0 0 6px}.case ol{margin:0;padding-left:22px}.top h2,h2{margin:0 0 8px;font-size:1.26rem}.check{margin-top:14px}.check li{margin:.4em 0}.route{fill:none;stroke-linecap:round;stroke-linejoin:round}.legend{font-size:.875rem;margin-top:10px}.tag{font-size:.85rem;color:#ffc479}.status{color:#ffc479}.tools{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}@media(max-width:850px){main{padding:15px}.grid{grid-template-columns:1fr}} 
</style><main><div class="top"><div><h1>타슈 재배치 시나리오</h1><p class="muted">2026년 11월 10일 관측 기록으로 계산한 작업 계획 · 한국 시각</p></div></div>
<div id="controls" class="controls" aria-label="시나리오 선택"></div><div class="grid"><section class="panel"><svg id="map" viewBox="0 0 850 530" class="map" role="img" aria-label="대전 대여소 및 계산된 도로 경로"></svg><p id="mapFoot" class="legend muted"></p><div class="tools"><a id="download" class="download" download="route.geojson">선택한 경로 GeoJSON</a></div></section><section class="panel"><div id="metrics" aria-live="polite"></div><div id="orders"></div></section></div>
<section class="panel check"><h2 id="reviewTitle"></h2><ol id="checks"></ol><p class="muted">차량 적재량, 상하차 수량, 작업·복귀 마감은 코드로 검사했습니다.</p></section></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const {plans,review,network}=JSON.parse(document.querySelector('#payload').textContent);
const cases=new Map(plans.cases.map(x=>[x.id,x]));const svg=document.querySelector('#map');const els=id=>document.getElementById(id);
const colors=['#61e6ce','#87bdfd','#ffc479'];let url;
const clock=s=>s?.slice(11,19)??'—';const d=(tag,text,cl)=>{const n=document.createElement(tag);n.textContent=text;if(cl)n.className=cl;return n;};
const g=(tag,attrs)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);svg.append(n);return n;};
const button=(name,id)=>{const n=d('button',name);n.dataset.case=id;n.type='button';n.onclick=()=>show(id);els('controls').append(n);};
if(review.simultaneous)button('3개 권역 함께 보기','fleet');for(const c of cases.values())button(c.name,c.id);
function show(id){
 const chosen=id==='fleet'?review.fleet.map(k=>cases.get(k)):[cases.get(id)];const routes=chosen.filter(c=>c.status==='plan');
 for(const b of els('controls').children)b.setAttribute('aria-pressed',b.dataset.case===id);
 const coordinates=routes.flatMap(c=>c.geometry.coordinates);const reference=coordinates.length?coordinates:network.boundary.flat();
 const xs=reference.map(p=>p[0]);const ys=reference.map(p=>p[1]);const cos=.806, cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2;
 const scale=Math.min(680/Math.max(.002,(Math.max(...xs)-Math.min(...xs))*cos),400/Math.max(.002,Math.max(...ys)-Math.min(...ys)));
 const pt=p=>[425+(p[0]-cx)*scale*cos,265-(p[1]-cy)*scale];const poly=(points,color,width)=>g('polyline',{points:points.map(p=>pt(p).join(',')).join(' '),stroke:color,'stroke-width':width,class:'route'});
 svg.replaceChildren();for(const b of network.boundary)poly(b,'#426078',1.5);
 for(const p of network.stations){const [x,y]=pt(p);if(x>5&&y>5&&x<845&&y<525)g('circle',{cx:x,cy:y,r:2,fill:'#43617c'});}
 let features=[];for(const [i,c] of routes.entries()){
   const color=colors[i%colors.length];poly(c.geometry.coordinates,color,5);
   features.push({type:'Feature',geometry:c.geometry,properties:{case:c.name,kind:'vehicle_route'}});
   c.stops.forEach((s,j)=>{const [x,y]=pt([s.lon,s.lat]);g('circle',{cx:x,cy:y,r:13,fill:'#0d1827',stroke:color,'stroke-width':2.5});
     g('text',{x,y:y+4,fill:'#f1f6fd','text-anchor':'middle','font-size':12}).textContent=j+1;
     features.push({type:'Feature',geometry:{type:'Point',coordinates:[s.lon,s.lat]},properties:{case:c.name,name:s.name,action:s.action,bikes:s.bikes,arrival:s.arrival,completion:s.completion}});});
 }
 if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(new Blob([JSON.stringify({type:'FeatureCollection',features},null,2)],{type:'application/geo+json'}));
 els('download').href=url;els('download').hidden=!routes.length;
 els('mapFoot').textContent='색 선은 차량의 왕복 도로 경로입니다. 번호는 작업 순서입니다. · © OpenStreetMap contributors';
 els('metrics').replaceChildren(d('div',routes.reduce((a,c)=>a+c.moved_bikes,0)+'대 이동 계획','metric'),d('p',id==='fleet'?'차량 3대가 각 권역에서 출발하는 가정':'선택한 관측과 작업 조건 적용','muted'));
 els('orders').replaceChildren();for(const [i,c] of chosen.entries()){
   const row=d('article','','case');const title=d('h3',c.name);title.style.color=colors[i%colors.length];row.append(title);
   if(c.status!=='plan'){row.append(d('p',c.reason,'status'));els('orders').append(row);continue;}
   row.append(d('p',`${c.moved_bikes}대 · 왕복 ${c.distance_km.toFixed(2)}km · ${c.work_minutes.toFixed(1)}분`));
   const steps=d('ol','');for(const s of c.stops)steps.append(d('li',`${clock(s.completion)} ${s.name} · ${s.action==='pickup'?'수거':'배치'} ${s.bikes}대`));row.append(steps);
   row.append(d('p',`복귀 ${clock(c.returned)} · 배치 마감 여유 ${review.checks[c.id].work_slack_seconds}초`,'tag'));els('orders').append(row);
 }
}
els('reviewTitle').textContent=review.mode==='llm_priority'?'LLM이 정리한 출발 전 확인 순서':'출발 전 확인 순서';
for(const fact of review.selected)els('checks').append(d('li',fact.text));show(review.simultaneous?'fleet':plans.cases[0].id);
</script></html>'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    def read(name):return json.loads((a.data/name).read_bytes())
    full=read('city-network.json')
    plans=read('plans.json')
    case_fields=('id','name','status','reason','geometry','moved_bikes','distance_km','work_minutes','returned')
    stop_fields=('name','action','bikes','completion','arrival','lon','lat')
    public_cases=[dict({k:case[k] for k in case_fields if k in case},
                       stops=[{k:step[k] for k in stop_fields} for step in case['stops']]) for case in plans['cases']]
    value=dict(plans=dict(cases=public_cases),review=read('review.json'),
               network=dict(boundary=full['boundary'],stations=[[round(n,4) for n in pair] for pair in full['stations']]))
    a.out.write_text(HTML.replace('__PAYLOAD__',json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')),encoding='utf-8')
