'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):'미관측';
const pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'미관측';
const time=s=>new Date(s).toLocaleTimeString('ko-KR',{timeZone:'Asia/Seoul',hour:'2-digit',minute:'2-digit',hour12:false});
let data,model='recent_cfm',selected,page=0,stage=0,playTimer=null,routeTimer=null,routeIndex=-1;
const names={recent_cfm:'GCN + Flow Matching',recent_poisson:'GCN + Poisson',persistence:'현재 재고 유지',recent_trend:'최근 변화 연장',no_recent_poisson:'최근 입력 없는 GCN'};
const moneyless=v=>Number.isFinite(v)?v.toLocaleString('ko-KR'):'—';
function project(points,w,h,pad=35){
 const x=points.map(p=>p.lon),y=points.map(p=>p.lat),minX=Math.min(...x),maxX=Math.max(...x),minY=Math.min(...y),maxY=Math.max(...y),mid=(minY+maxY)/2;
 const correction=Math.cos(mid*Math.PI/180),dx=(maxX-minX)*correction||.001,dy=maxY-minY||.001,scale=Math.min((w-2*pad)/dx,(h-2*pad)/dy);
 return p=>[w/2+(p.lon-(minX+maxX)/2)*correction*scale,h/2-(p.lat-(minY+maxY)/2)*scale];
}
function grid(w,h){return `<defs><pattern id="grid${w}${h}" width="35" height="35" patternUnits="userSpaceOnUse"><path d="M35 0H0V35" fill="none" stroke="#dfe8e3" stroke-width=".6"/></pattern></defs><rect width="${w}" height="${h}" fill="url(#grid${w}${h})"/><text x="20" y="25" fill="#91a49b" font-size="10">N ↑</text>`;}
function selectStation(sid,scroll=false){
 selected=data.stations.find(s=>s.id===sid);if(!selected)return;const f=selected.forecast[model];
 $('stationId').textContent=selected.id;$('stationName').textContent=selected.name;
 $('stationContext').textContent='2026.10.31 '+time(data.episode.scheduled_at_utc)+' → '+time(data.episode.target_at_utc)+' · '+names[model];
 $('stationCurrent').textContent=selected.current+'대';$('stationPrediction').textContent=f.median+'대';$('stationTruth').textContent=selected.truth===null?'미관측':selected.truth+'대';
 $('stationInterval').textContent=`${f.low}–${f.high}대`;$('stationP0').textContent=pct(f.p0);
 const values=[selected.current,f.median,selected.truth],max=Math.max(1,...values.filter(v=>v!==null),f.high),colors=['#a1b7af','#168769','#416279'];
 $('comparison').innerHTML=`<line x1="20" y1="131" x2="380" y2="131" stroke="#dde6e1"/>`+values.map((v,i)=>{const x=65+i*130,h=(v??0)/max*92;return `<rect x="${x-22}" y="${130-h}" width="44" height="${Math.max(v===null?0:2,h)}" rx="4" fill="${colors[i]}"/><text x="${x}" y="${Math.max(16,122-h)}" text-anchor="middle" font-size="12" fill="${colors[i]}">${v===null?'미관측':v+'대'}</text><text x="${x}" y="155" text-anchor="middle" font-size="11" fill="#83958c">${['현재 관측','예측 중앙값','미래 실측'][i]}</text>`;}).join('');
 drawStations();if(scroll)$('stationName').scrollIntoView({block:'center',behavior:'smooth'});
}
function drawStations(){
 const values=data.stations.filter(s=>s.lat!==null),point=project(values,680,390);
 $('stationMap').innerHTML=grid(680,390)+values.map(s=>{const [x,y]=point(s);return `<circle data-id="${esc(s.id)}" cx="${x}" cy="${y}" r="${s.id===selected?.id?6:2.7}" fill="${s.current===0?'#c88f4e':'#4b9d81'}" opacity="${s.id===selected?.id?1:.65}" ${s.id===selected?.id?'stroke="#164936" stroke-width="2"':''}><title>${esc(s.name)} · 현재 ${s.current}대</title></circle>`;}).join('');
 $('stationMap').querySelectorAll('[data-id]').forEach(c=>c.onclick=()=>selectStation(c.dataset.id));
}
function table(){
 const q=$('search').value.trim().toLowerCase(),filter=$('filter').value;
 const rows=data.stations.filter(s=>(s.name.toLowerCase().includes(q)||s.id.toLowerCase().includes(q))&&(filter==='all'||filter==='changed'&&s.truth!==null&&s.truth!==s.current||filter==='error'&&s.truth!==null&&s.forecast[model].median!==s.truth||filter==='zero'&&s.current===0)).sort((a,b)=>a.display_rank-b.display_rank);
 const totalPages=Math.max(1,Math.ceil(rows.length/15));page=Math.min(page,totalPages-1);
 $('stationRows').innerHTML=rows.slice(page*15,page*15+15).map(s=>{const f=s.forecast[model],e=s.truth===null?null:Math.abs(f.median-s.truth);return `<tr><td><button class="station-link" data-id="${esc(s.id)}">${esc(s.name)}<small>${esc(s.id)}</small></button></td><td>${s.current}대</td><td>${f.median}대</td><td>${f.low}–${f.high}대</td><td>${pct(f.p0)}</td><td class="truth">${s.truth===null?'미관측':s.truth+'대'}</td><td class="${e?'error-cell':''}">${e===null?'—':e+'대'}</td></tr>`;}).join('')||'<tr><td colspan="7">검색 결과가 없습니다.</td></tr>';
 $('tableCount').textContent=moneyless(rows.length)+'곳';$('pageLabel').textContent=`${page+1} / ${totalPages} 페이지 · 15곳씩 표시`;
 $('pagePrev').disabled=page===0;$('pageNext').disabled=page>=totalPages-1;
 $('stationRows').querySelectorAll('button').forEach(b=>b.onclick=()=>selectStation(b.dataset.id,true));
}
function metricCards(){
 $('pairs').innerHTML=moneyless(data.episode.pairs)+'<small>곳</small>';
 $('mae').innerHTML=num(data.live_metrics[model].median_abs)+'<small>대</small>';
 $('moved').innerHTML=(data.shadow.total_moved_bikes??'—')+'<small>대</small>';
 $('routeMinutes').innerHTML=(data.shadow.route?num(data.shadow.route.return_minutes,1):'—')+'<small>분</small>';
 const all=data.september.groups.all;
 $('officialMetrics').innerHTML=[['전체 평가 표본',moneyless(data.september.coverage.valid_pairs)+'쌍'],['현재유지 · 재고 수량 MAE',num(all.persistence.median_mae,6)+'대'],['GCN + FM · 재고 수량 MAE',num(all.recent_cfm.median_mae,6)+'대'],['GCN + Poisson · 분포 fairCRPS',num(all.recent_poisson.fair_crps,6)],['GCN + FM · 분포 fairCRPS',num(all.recent_cfm.fair_crps,6)],['FM 추가 가치 가설','지지하지 않음']].map(([k,v])=>`<div class="score-row"><span>${k}</span><b>${v}</b></div>`).join('');
}
function setStage(n){
 stage=Math.max(0,Math.min(4,n));const e=data.episode;
 const steps=[['실제 API 관측',time(e.scheduled_at_utc)+' 현재 재고를 받았습니다.','대여소별 표시 재고와 관측 시각을 입력으로 사용했습니다. 누락을 0대로 채우지 않았습니다.'],['발행 당시 고정된 모델',time(e.issued_at_utc)+' 예측을 발행했습니다.','다음 10분의 재고 분포를 생성했습니다. 이후 관측은 아직 입력되지 않았고 API로 가중치를 갱신하지 않았습니다.'],['별도로 수집한 미래 정답',time(e.observed_future_at_utc)+' 실제 재고와 비교했습니다.',`${e.pairs.toLocaleString()}개 대여소가 대응됩니다. 아래 표에서 예측 중앙값·범위·실측·오차를 모두 볼 수 있습니다. 이 한 회차를 전체 정확도로 일반화하지 않습니다.`],['가정에 따른 그림자계획','실제 대여소에 수량과 방문 순서를 붙였습니다.','당시 예측과 현재 재고만으로 후보를 고른 뒤 계산했습니다. 가상 차량·거점·주차 조건과 OSRM 도로 추정에 따른 계획이며 실제 수행하지 않았습니다.'],['검증된 범위까지','예측 결과와 운영 효과를 구분합니다.','예측은 실제 API로 비교했습니다. 계획의 적재·시간 제약도 검사했습니다. 재배치 이후 현장 결과는 미관측이므로 개선 효과를 숫자로 만들지 않았습니다.']];
 const s=steps[stage];$('stepTag').textContent=s[0];$('stepTitle').textContent=s[1];$('stepText').textContent=s[2];$('stepCount').textContent=`${stage+1} / 5`;
 document.querySelectorAll('[data-step]').forEach((b,i)=>{b.classList.toggle('active',i===stage);b.classList.toggle('done',i<stage);});$('prev').disabled=stage===0;$('next').disabled=stage===4;
}
function stopPlay(){clearInterval(playTimer);playTimer=null;$('play').textContent='▶ 전체 과정 재생';}
function routeMap(index=-1){
 const s=data.shadow;if(!s.route)return;const locations=s.selected_stations,geo=(s.geometry?.coordinates||[]).map(p=>({lon:p[0],lat:p[1]})),point=project([...locations,...geo],680,400,55);
 let markup=grid(680,400);const lookup=Object.fromEntries(locations.map(s=>[s.id,s]));lookup.ASSUMED_DEPOT=s.assumed_depot;
 const points=geo.length?geo:s.route.sequence.map(k=>lookup[k]);
 markup+=`<polyline points="${points.map(p=>point(p).join(',')).join(' ')}" fill="none" stroke="#2a9270" stroke-width="4" stroke-linejoin="round" opacity=".6" ${geo.length?'':'stroke-dasharray="8 6"'}/>`;
 const stops=s.route.stops;
 stops.forEach((st,i)=>{const p=lookup[st.station_id],[x,y]=point(p);markup+=`<circle cx="${x}" cy="${y}" r="${i===index?18:14}" fill="${st.action==='pickup'?'#be914e':'#158163'}" stroke="white" stroke-width="3"/><text x="${x}" y="${y+4}" text-anchor="middle" font-size="12" fill="white" font-weight="700">${i+1}</text><text x="${x}" y="${y-22}" text-anchor="middle" font-size="11" fill="#456358">${esc(p.name.slice(0,17))}</text>`;});
 const [dx,dy]=point(s.assumed_depot);markup+=`<rect x="${dx-6}" y="${dy-7}" width="12" height="12" fill="#173c4d" stroke="white"/><text x="${dx+15}" y="${dy+22}" font-size="10" fill="#536b73">가상 출발·복귀 거점</text>`;
 $('routeMap').innerHTML=markup;
 document.querySelectorAll('.stop').forEach((el,i)=>el.classList.toggle('active',i===index));
 const load=index<0?0:stops[index]?.vehicle_load_after??0;$('loadText').textContent=`${load} / 6대`;$('loadBar').style.width=(load/6*100)+'%';
}
function planView(){
 const s=data.shadow;$('assumptions').innerHTML=['계획 입력: 당시 고정 GCN+FM','가상 차량 1대 · 적재 6대','가상 목표재고 2대','작업시간 0.1분/대 가정','거점: 선택 공급 대여소로 가정','허용 주차공간 미확인'].map(x=>`<span>${x}</span>`).join('');
 if(s.status==='blocked'){$('planContent').hidden=true;$('blocked').hidden=false;$('blocked').textContent='실제 대여소 계획 계산 중단: '+s.reason;return;}
 const lookup=Object.fromEntries(s.selected_stations.map(s=>[s.id,s]));
 $('routeStops').innerHTML=s.route.stops.map((st,i)=>`<div class="stop" tabindex="0" role="button" data-index="${i}"><span class="stop-number">${i+1}</span><div><div class="stop-name">${esc(lookup[st.station_id].name)}</div><small>출발 ${num(st.eta_minutes,1)}분 후 도착 · ${num(st.completion_minutes,1)}분에 작업 완료</small></div><span class="stop-action ${st.action==='pickup'?'pickup':'deliver'}">${st.action==='pickup'?'수거':'배치'} ${st.bikes}대<small>적재 ${st.vehicle_load_after}대</small></span></div>`).join('')||'<p class="fine">제약 안에서 선택된 이동이 없습니다.</p>';
 $('routeStops').querySelectorAll('.stop').forEach(b=>{const choose=()=>{routeIndex=+b.dataset.index;routeMap(routeIndex);};b.onclick=choose;b.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};});
 $('returnTime').textContent=`↩ 가상 거점 복귀 ${num(s.route.return_minutes,2)}분 · 최대 적재 ${s.route.peak_load}대`;
 $('roadCaption').textContent=s.geometry?'OSRM car 도로 추정 · 실측 교통·차량 진입 검증 아님 · 거점 가정':'도로시간은 OSRM 추정 · 선은 방문 순서 연결이며 도로 경로 아님';
 $('transfers').innerHTML=s.transfers.map(t=>`<div class="transfer"><span>${esc(lookup[t.origin].name)}<small>수거 · ${esc(t.origin)}</small></span><span class="arrow">→</span><span>${esc(lookup[t.destination].name)}<small>배치 · ${esc(t.destination)}</small></span><b>${t.bikes}대 이동</b></div>`).join('')||'<p class="fine">실행 가능한 이동을 선택하지 않았습니다.</p>';
 routeMap();$('animateRoute').onclick=()=>{clearInterval(routeTimer);routeIndex=-1;routeMap();routeTimer=setInterval(()=>{routeIndex++;if(routeIndex>=s.route.stops.length){clearInterval(routeTimer);routeMap(-1);return;}routeMap(routeIndex);},1000);};
}
function download(){const blob=new Blob([JSON.stringify({episode:data.episode,metrics:data.live_metrics,september:data.september,shadow:data.shadow,limitations:data.limitations,provenance:data.provenance},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='tashu-recorded-results-and-shadow-plan.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function init(){
 try{const r=await fetch('/api/results',{cache:'no-store'});if(!r.ok)throw Error('결과 자료를 불러올 수 없습니다.');data=await r.json();if(data.hypothesis_id!=='H-0016'||!data.not_live||!data.stations?.length)throw Error('검증된 기록 형식이 아닙니다.');
 $('episodeTime').textContent=`2026.10.31 ${time(data.episode.scheduled_at_utc)} → ${time(data.episode.target_at_utc)}`;
 $('loading').hidden=true;$('content').hidden=false;$('play').disabled=false;
 metricCards();table();planView();setStage(0);selectStation(data.shadow.selected_stations?.find(s=>s.current<2)?.id||data.stations[0].id);
 $('model').onchange=()=>{model=$('model').value;page=0;metricCards();table();selectStation(selected.id);};
 $('search').oninput=$('filter').onchange=()=>{page=0;table();};$('pagePrev').onclick=()=>{page--;table();};$('pageNext').onclick=()=>{page++;table();};
 $('prev').onclick=()=>{stopPlay();setStage(stage-1);};$('next').onclick=()=>{stopPlay();setStage(stage+1);};document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{stopPlay();setStage(+b.dataset.step);});
 $('play').onclick=()=>{if(playTimer){stopPlay();return;}setStage(0);$('play').textContent='Ⅱ 재생 중지';playTimer=setInterval(()=>{if(stage===4){stopPlay();return;}setStage(stage+1);if(stage===2)$('validation').scrollIntoView({behavior:'smooth'});if(stage===3)$('planning').scrollIntoView({behavior:'smooth'});if(stage===4)$('research').scrollIntoView({behavior:'smooth'});},4500);};$('download').onclick=download;
 }catch(e){$('loading').hidden=true;$('error').hidden=false;$('error').textContent=e.message+' 비공개 결과 묶음과 화면 연결을 확인해 주세요. 예시 값을 대신 표시하지 않았습니다.';}
}
init();
